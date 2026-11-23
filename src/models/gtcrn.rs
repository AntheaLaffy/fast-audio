//! Pinned GTCRN architecture (upstream 502ebfab64da7c4a9af78dcb9c6ceef1ebb01c73).
//! Parameters are named FP32 tensors, independent of PyTorch's checkpoint container.
use crate::{
    Activation, Conv2dSpec, GruSpec, GruWeights, Result, Tensor,
    error::invalid,
    graph::{Builder, Program, Unary, Value},
};
use std::collections::BTreeMap;

pub type Parameters = BTreeMap<String, Tensor>;
pub struct ModelGraph {
    pub program: Program,
    pub probes: BTreeMap<String, Value>,
}
fn get<'a>(p: &'a Parameters, name: &str) -> Result<&'a Tensor> {
    p.get(name)
        .ok_or_else(|| invalid(format!("missing GTCRN parameter {name}")))
}
fn param<'a>(p: &'a Parameters, name: &str, suffix: &str) -> Result<&'a Tensor> {
    get(p, &format!("{name}.{suffix}"))
}

fn layout(g: &mut Builder, x: Value, axis: &[usize], output: Option<&[usize]>) -> Result<Value> {
    let shape = g.shape(x)?.to_vec();
    let ranges: Vec<_> = shape.iter().map(|&n| (0, n)).collect();
    let output = output
        .map(<[usize]>::to_vec)
        .unwrap_or_else(|| axis.iter().map(|&a| shape[a]).collect());
    g.layout(x, &ranges, axis, output)
}
fn part(
    g: &mut Builder,
    x: Value,
    axis: usize,
    start: usize,
    len: usize,
    output: Option<&[usize]>,
) -> Result<Value> {
    let mut shape = g.shape(x)?.to_vec();
    let mut ranges: Vec<_> = shape.iter().map(|&n| (0, n)).collect();
    ranges[axis] = (start, len);
    shape[axis] = len;
    let axes: Vec<_> = (0..shape.len()).collect();
    g.layout(x, &ranges, &axes, output.unwrap_or(&shape))
}
fn linear(
    g: &mut Builder,
    p: &Parameters,
    name: &str,
    x: Value,
    output: Option<&[usize]>,
) -> Result<Value> {
    let w = param(p, name, "weight")?;
    let b = p.get(&format!("{name}.bias"));
    let mut shape = g.shape(x)?.to_vec();
    *shape.last_mut().unwrap() = w.shape().dims()[0];
    g.linear(x, w, b, output.unwrap_or(&shape))
}
fn gru_weights(p: &Parameters, name: &str, reverse: bool) -> Result<GruWeights> {
    let wh = param(p, name, "weight_hh_l0")?;
    let wi = param(p, name, "weight_ih_l0")?;
    let &[gates, hidden] = wh.shape().dims() else {
        return Err(invalid("GTCRN recurrent weights must be matrices"));
    };
    let &[input_gates, input] = wi.shape().dims() else {
        return Err(invalid("GTCRN input weights must be matrices"));
    };
    if hidden.checked_mul(3) != Some(gates) || gates != input_gates {
        return Err(invalid("GTCRN gate dimensions differ"));
    }
    let directions = if reverse { 2 } else { 1 };
    let stack = |base: &str, shape: &[usize]| -> Result<Tensor> {
        let mut data = param(p, name, &format!("{base}_l0"))?.data().to_vec();
        if reverse {
            data.extend_from_slice(param(p, name, &format!("{base}_l0_reverse"))?.data());
        }
        Tensor::from_vec(shape, data)
    };
    Ok(GruWeights {
        input: stack("weight_ih", &[directions, 3 * hidden, input])?,
        hidden: stack("weight_hh", &[directions, 3 * hidden, hidden])?,
        input_bias: stack("bias_ih", &[directions, 3 * hidden])?,
        hidden_bias: stack("bias_hh", &[directions, 3 * hidden])?,
    })
}
fn gru(
    g: &mut Builder,
    p: &Parameters,
    name: &str,
    x: Value,
    bidirectional: bool,
) -> Result<Value> {
    let &[batch, steps, input_size] = g.shape(x)? else {
        return Err(invalid("GTCRN recurrent layout must be batch-first"));
    };
    let w = gru_weights(p, name, bidirectional)?;
    let spec = GruSpec {
        batch,
        steps,
        input_size,
        hidden_size: w.hidden.shape().dims()[2],
        bidirectional,
    };
    let initial = g.constant(Tensor::zeros(spec.state_shape())?)?;
    Ok(g.gru(x, initial, spec, w)?.0)
}
fn grnn(
    g: &mut Builder,
    p: &Parameters,
    name: &str,
    x: Value,
    bidirectional: bool,
) -> Result<Value> {
    let shape = g.shape(x)?.to_vec();
    let rank = shape.len();
    let input = shape[rank - 1];
    let steps = shape[rank - 2];
    let batch = shape[..rank - 2].iter().product::<usize>();
    if input % 2 != 0 {
        return Err(invalid("grouped GTCRN RNN needs even features"));
    }
    let a = part(
        g,
        x,
        rank - 1,
        0,
        input / 2,
        Some(&[batch, steps, input / 2]),
    )?;
    let b = part(
        g,
        x,
        rank - 1,
        input / 2,
        input / 2,
        Some(&[batch, steps, input / 2]),
    )?;
    let a = gru(g, p, &format!("{name}.rnn1"), a, bidirectional)?;
    let b = gru(g, p, &format!("{name}.rnn2"), b, bidirectional)?;
    g.concat(a, b, 2, false)
}

/// Fold inference BatchNorm into either OIHW or grouped IOHW parameters once.
fn folded(
    p: &Parameters,
    conv: &str,
    bn: &str,
    transposed: bool,
    groups: usize,
) -> Result<(Tensor, Tensor)> {
    let source = param(p, conv, "weight")?;
    let &[rows, columns, kh, kw] = source.shape().dims() else {
        return Err(invalid("GTCRN convolution parameters must be rank four"));
    };
    let cout = if transposed {
        columns
            .checked_mul(groups)
            .ok_or_else(|| invalid("GTCRN channel count overflows"))?
    } else {
        rows
    };
    let gamma = param(p, bn, "weight")?;
    let beta = param(p, bn, "bias")?;
    let mean = param(p, bn, "running_mean")?;
    let variance = param(p, bn, "running_var")?;
    for t in [gamma, beta, mean, variance] {
        t.shape().expect(&[cout])?;
    }
    if groups == 0
        || rows % groups != 0
        || variance.data().iter().any(|&v| !v.is_finite() || v < 0.0)
    {
        return Err(invalid("invalid GTCRN BatchNorm/group parameters"));
    }
    let scale: Vec<_> = gamma
        .data()
        .iter()
        .zip(variance.data())
        .map(|(&g, &v)| g / (v + 1e-5).sqrt())
        .collect();
    let mut w = source.clone();
    for (i, v) in w.data_mut().iter_mut().enumerate() {
        let channel = if transposed {
            (i / (columns * kh * kw)) / (rows / groups) * columns + i / (kh * kw) % columns
        } else {
            i / (columns * kh * kw)
        };
        *v *= scale[channel];
    }
    let bias = p.get(&format!("{conv}.bias"));
    if let Some(b) = bias {
        b.shape().expect(&[cout])?;
    }
    let bias = (0..cout)
        .map(|c| (bias.map_or(0.0, |b| b.data()[c]) - mean.data()[c]) * scale[c] + beta.data()[c])
        .collect();
    Ok((w, Tensor::from_vec([cout], bias)?))
}
#[derive(Clone, Copy)]
struct Conv {
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    groups: usize,
    transposed: bool,
}
impl Default for Conv {
    fn default() -> Self {
        Self {
            stride: [1, 1],
            padding: [0, 0],
            dilation: [1, 1],
            groups: 1,
            transposed: false,
        }
    }
}
fn conv_bn(
    g: &mut Builder,
    p: &Parameters,
    x: Value,
    conv: &str,
    bn: &str,
    options: Conv,
) -> Result<Value> {
    let (w, b) = folded(p, conv, bn, options.transposed, options.groups)?;
    let &[batch, in_channels, h, width] = g.shape(x)? else {
        return Err(invalid("GTCRN convolution input must be NCHW"));
    };
    let shape = w.shape().dims();
    let spec = Conv2dSpec {
        batch,
        in_channels,
        out_channels: b.shape().len(),
        input: [h, width],
        kernel: [shape[2], shape[3]],
        stride: options.stride,
        padding: options.padding,
        dilation: options.dilation,
        groups: options.groups,
        transposed: options.transposed,
        output_padding: [0, 0],
    };
    g.conv2d(x, spec, &w, Some(&b), Activation::Identity)
}
fn prelu(g: &mut Builder, p: &Parameters, name: &str, x: Value) -> Result<Value> {
    let slope = param(p, name, "weight")?;
    slope.shape().expect(&[1])?;
    g.unary(x, Unary::PRelu(slope.data()[0]))
}
fn conv_block(
    g: &mut Builder,
    p: &Parameters,
    name: &str,
    x: Value,
    groups: usize,
    transposed: bool,
    last: bool,
) -> Result<Value> {
    let x = conv_bn(
        g,
        p,
        x,
        &format!("{name}.conv"),
        &format!("{name}.bn"),
        Conv {
            stride: [1, 2],
            padding: [0, 2],
            groups,
            transposed,
            ..Conv::default()
        },
    )?;
    if last {
        g.unary(x, Unary::Tanh)
    } else {
        prelu(g, p, &format!("{name}.act"), x)
    }
}
fn attention(g: &mut Builder, p: &Parameters, name: &str, x: Value) -> Result<Value> {
    let &[b, c, t, _] = g.shape(x)? else {
        return Err(invalid("GTCRN attention needs NCHW"));
    };
    let energy = g.mean_square_last(x)?;
    let energy = layout(g, energy, &[0, 2, 1], None)?;
    let at = gru(g, p, &format!("{name}.att_gru"), energy, false)?;
    let at = linear(g, p, &format!("{name}.att_fc"), at, None)?;
    let at = g.unary(at, Unary::Sigmoid)?;
    let at = layout(g, at, &[0, 2, 1], Some(&[b, c, t, 1]))?;
    g.multiply_broadcast(x, at)
}
fn gt_block(
    g: &mut Builder,
    p: &Parameters,
    name: &str,
    x: Value,
    dilation: usize,
    transposed: bool,
) -> Result<Value> {
    let channels = g.shape(x)?[1];
    let x1 = part(g, x, 1, 0, channels / 2, None)?;
    let x2 = part(g, x, 1, channels / 2, channels / 2, None)?;
    let h = g.unfold_frequency(x1, 3)?;
    let options = Conv {
        transposed,
        ..Conv::default()
    };
    let h = conv_bn(
        g,
        p,
        h,
        &format!("{name}.point_conv1"),
        &format!("{name}.point_bn1"),
        options,
    )?;
    let h = prelu(g, p, &format!("{name}.point_act"), h)?;
    let h = g.pad_axis(h, 2, 2 * dilation, 0)?;
    let depth = Conv {
        dilation: [dilation, 1],
        padding: [if transposed { 2 * dilation } else { 0 }, 1],
        groups: 16,
        ..options
    };
    let h = conv_bn(
        g,
        p,
        h,
        &format!("{name}.depth_conv"),
        &format!("{name}.depth_bn"),
        depth,
    )?;
    let h = prelu(g, p, &format!("{name}.depth_act"), h)?;
    let h = conv_bn(
        g,
        p,
        h,
        &format!("{name}.point_conv2"),
        &format!("{name}.point_bn2"),
        options,
    )?;
    let h = attention(g, p, &format!("{name}.tra"), h)?;
    g.concat(h, x2, 1, true)
}
fn dual_path(g: &mut Builder, p: &Parameters, name: &str, x: Value) -> Result<Value> {
    let &[b, c, t, f] = g.shape(x)? else {
        return Err(invalid("GTCRN dual path needs NCHW"));
    };
    let residual = layout(g, x, &[0, 2, 3, 1], None)?;
    let intra = grnn(g, p, &format!("{name}.intra_rnn"), residual, true)?;
    let intra = linear(
        g,
        p,
        &format!("{name}.intra_fc"),
        intra,
        Some(&[b, t, f, c]),
    )?;
    let intra = g.layer_norm(
        intra,
        param(p, &format!("{name}.intra_ln"), "weight")?,
        param(p, &format!("{name}.intra_ln"), "bias")?,
        1e-8,
    )?;
    let intra = g.add(residual, intra)?;
    let inter = layout(g, intra, &[0, 2, 1, 3], None)?;
    let inter = grnn(g, p, &format!("{name}.inter_rnn"), inter, false)?;
    let inter = linear(
        g,
        p,
        &format!("{name}.inter_fc"),
        inter,
        Some(&[b, f, t, c]),
    )?;
    let inter = layout(g, inter, &[0, 2, 1, 3], None)?;
    let inter = g.layer_norm(
        inter,
        param(p, &format!("{name}.inter_ln"), "weight")?,
        param(p, &format!("{name}.inter_ln"), "bias")?,
        1e-8,
    )?;
    let out = g.add(intra, inter)?;
    layout(g, out, &[0, 3, 1, 2], None)
}
fn erb(g: &mut Builder, p: &Parameters, x: Value, synthesis: bool) -> Result<Value> {
    let f = g.shape(x)?[3];
    let low = part(g, x, 3, 0, 65, None)?;
    let high = part(g, x, 3, 65, f - 65, None)?;
    let high = linear(
        g,
        p,
        if synthesis {
            "erb.ierb_fc"
        } else {
            "erb.erb_fc"
        },
        high,
        None,
    )?;
    g.concat(low, high, 3, false)
}

/// Offline spectrum-to-spectrum model. Each invocation starts all recurrent states at zero.
/// Batch and frame counts are fixed. Spectra use the library's `[B,T,257,2]` layout.
pub fn build(batch: usize, frames: usize, parameters: &Parameters) -> Result<ModelGraph> {
    let mut g = Builder::new([batch, frames, 257, 2])?;
    let input = g.input();
    let p = parameters;
    let mut probes = BTreeMap::new();
    let mut x = g.complex_features(input, 1e-12)?;
    x = erb(&mut g, p, x, false)?;
    probes.insert("erb.bm".into(), x);
    x = g.unfold_frequency(x, 3)?;
    probes.insert("sfe".into(), x);
    let mut skips = Vec::new();
    for i in 0..5 {
        let name = format!("encoder.en_convs.{i}");
        x = if i < 2 {
            conv_block(
                &mut g,
                p,
                &name,
                x,
                if i == 0 { 1 } else { 2 },
                false,
                false,
            )?
        } else {
            gt_block(&mut g, p, &name, x, [1, 2, 5][i - 2], false)?
        };
        skips.push(x);
        probes.insert(name, x);
    }
    for name in ["dpgrnn1", "dpgrnn2"] {
        x = dual_path(&mut g, p, name, x)?;
        probes.insert(name.into(), x);
    }
    for i in 0..5 {
        let name = format!("decoder.de_convs.{i}");
        x = g.add(x, skips[4 - i])?;
        x = if i < 3 {
            gt_block(&mut g, p, &name, x, [5, 2, 1][i], true)?
        } else {
            conv_block(
                &mut g,
                p,
                &name,
                x,
                if i == 3 { 2 } else { 1 },
                true,
                i == 4,
            )?
        };
        probes.insert(name, x);
    }
    x = erb(&mut g, p, x, true)?;
    probes.insert("erb.bs".into(), x);
    let out = g.complex_mask(x, input)?;
    Ok(ModelGraph {
        program: g.finish(out)?,
        probes,
    })
}

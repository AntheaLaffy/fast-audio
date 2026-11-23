use fast_audio::{
    Activation, Conv2dSpec, Tensor,
    cpu::{Conv2d, Cpu},
};
mod support;
use support::{backends, close, data, no_allocations};

fn cases() -> Vec<Conv2dSpec> {
    let s = Conv2dSpec {
        batch: 2,
        in_channels: 4,
        out_channels: 6,
        input: [5, 7],
        kernel: [2, 3],
        stride: [2, 1],
        padding: [1, 2],
        dilation: [1, 2],
        groups: 2,
        transposed: false,
        output_padding: [0, 0],
    };
    vec![
        s,
        Conv2dSpec { groups: 1, ..s },
        Conv2dSpec {
            in_channels: 3,
            out_channels: 3,
            groups: 3,
            kernel: [3, 3],
            stride: [1, 1],
            dilation: [2, 1],
            padding: [2, 1],
            ..s
        },
        Conv2dSpec {
            transposed: true,
            output_padding: [1, 0],
            ..s
        },
        Conv2dSpec {
            transposed: true,
            stride: [2, 2],
            padding: [0, 0],
            output_padding: [1, 1],
            ..s
        },
        Conv2dSpec {
            transposed: true,
            stride: [1, 1],
            dilation: [2, 2],
            padding: [0, 1],
            output_padding: [1, 1],
            ..s
        },
        Conv2dSpec {
            kernel: [1, 1],
            stride: [1, 1],
            padding: [0, 0],
            dilation: [1, 1],
            ..s
        },
        Conv2dSpec {
            kernel: [1, 1],
            stride: [1, 1],
            padding: [0, 0],
            dilation: [1, 1],
            transposed: true,
            ..s
        },
    ]
}
fn oracle(s: Conv2dSpec, x: &Tensor, w: &Tensor, bias: Option<&Tensor>, relu: bool) -> Vec<f64> {
    let o = s.output_shape().unwrap();
    let ci = s.in_channels / s.groups;
    let co = s.out_channels / s.groups;
    let mut result = vec![0.0; o.iter().product()];
    for n in 0..s.batch {
        for oc in 0..s.out_channels {
            for y in 0..o[2] {
                for z in 0..o[3] {
                    let mut value = bias.map_or(0.0, |b| b.data()[oc] as f64);
                    let group = oc / co;
                    for channel in 0..ci {
                        let ic = group * ci + channel;
                        for kh in 0..s.kernel[0] {
                            for kw in 0..s.kernel[1] {
                                let (ih, iw) = if s.transposed {
                                    let h = y as i64 + s.padding[0] as i64
                                        - (kh * s.dilation[0]) as i64;
                                    let w = z as i64 + s.padding[1] as i64
                                        - (kw * s.dilation[1]) as i64;
                                    if h < 0
                                        || w < 0
                                        || h % s.stride[0] as i64 != 0
                                        || w % s.stride[1] as i64 != 0
                                    {
                                        continue;
                                    }
                                    (h / s.stride[0] as i64, w / s.stride[1] as i64)
                                } else {
                                    (
                                        (y * s.stride[0] + kh * s.dilation[0]) as i64
                                            - s.padding[0] as i64,
                                        (z * s.stride[1] + kw * s.dilation[1]) as i64
                                            - s.padding[1] as i64,
                                    )
                                };
                                if ih < 0
                                    || iw < 0
                                    || ih >= s.input[0] as i64
                                    || iw >= s.input[1] as i64
                                {
                                    continue;
                                }
                                let index = if s.transposed {
                                    ((ic * co + oc % co) * s.kernel[0] + kh) * s.kernel[1] + kw
                                } else {
                                    ((oc * ci + channel) * s.kernel[0] + kh) * s.kernel[1] + kw
                                };
                                value += x.data()[((n * s.in_channels + ic) * s.input[0]
                                    + ih as usize)
                                    * s.input[1]
                                    + iw as usize] as f64
                                    * w.data()[index] as f64;
                            }
                        }
                    }
                    if relu {
                        value = value.max(0.0);
                    }
                    result[((n * s.out_channels + oc) * o[2] + y) * o[3] + z] = value;
                }
            }
        }
    }
    result
}

#[test]
fn prepared_forward_and_transposed_2d_match_f64_and_reuse_storage() {
    for cpu in backends() {
        for s in cases() {
            for has_bias in [false, true] {
                for relu in [false, true] {
                    let x = data(s.input_shape());
                    let mut weights = data(s.weight_shape());
                    let mut bias = has_bias.then(|| data([s.out_channels]));
                    let expected = oracle(s, &x, &weights, bias.as_ref(), relu);
                    let mut plan = Conv2d::new(
                        cpu,
                        s,
                        &weights,
                        bias.as_ref(),
                        if relu {
                            Activation::Relu
                        } else {
                            Activation::Identity
                        },
                    )
                    .unwrap_or_else(|e| panic!("{cpu:?} {s:?} bias={has_bias} relu={relu}: {e}"));
                    let mut out = Tensor::zeros(s.output_shape().unwrap()).unwrap();
                    weights.data_mut().fill(77.0);
                    if let Some(b) = &mut bias {
                        b.data_mut().fill(-33.0);
                    }
                    for _ in 0..2 {
                        plan.run(&x, &mut out).unwrap();
                    }
                    close(out.data(), &expected, 5e-6, 3e-5);
                    no_allocations(|| plan.run(&x, &mut out).unwrap());
                }
            }
        }
    }
}

#[test]
fn convolution_2d_rejects_geometry_and_wrong_equal_size_shapes() {
    let s = cases()[0];
    for bad in [
        Conv2dSpec { groups: 0, ..s },
        Conv2dSpec { groups: 3, ..s },
        Conv2dSpec {
            stride: [0, 1],
            ..s
        },
        Conv2dSpec {
            input: [usize::MAX, 2],
            ..s
        },
        Conv2dSpec {
            kernel: [0, 3],
            ..s
        },
        Conv2dSpec {
            output_padding: [1, 0],
            ..s
        },
        Conv2dSpec {
            transposed: true,
            output_padding: [2, 2],
            dilation: [1, 1],
            ..s
        },
    ] {
        assert!(bad.output_shape().is_err());
    }
    let x = data(s.input_shape());
    let weights = data(s.weight_shape());
    let mut plan = Conv2d::new(Cpu::Reference, s, &weights, None, Activation::Identity).unwrap();
    let mut out = Tensor::zeros(s.output_shape().unwrap()).unwrap();
    let wrong = Tensor::from_vec([x.shape().len()], x.data().to_vec()).unwrap();
    assert!(plan.run(&wrong, &mut out).is_err());
    let mut wrong = Tensor::zeros([out.shape().len()]).unwrap();
    assert!(plan.run(&x, &mut wrong).is_err());
    assert!(
        Conv2d::new(
            Cpu::Reference,
            s,
            &data([weights.shape().len()]),
            None,
            Activation::Identity
        )
        .is_err()
    );
}

#[cfg(feature = "cudnn")]
#[test]
fn cudnn_2d_graphs_match_f64_preserve_snapshots_and_rebind() {
    use fast_audio::cuda;
    for policy in [
        cuda::ConvolutionPolicy::AllowNonDeterministic,
        cuda::ConvolutionPolicy::RequireDeterministic,
    ] {
        let gpu = cuda::Cuda::with_convolution_policy(0, policy).unwrap();
        for s in cases() {
            for has_bias in [false, true] {
                for relu in [false, true] {
                    let x = data(s.input_shape());
                    let mut weights = data(s.weight_shape());
                    let mut bias = has_bias.then(|| data([s.out_channels]));
                    let expected = oracle(s, &x, &weights, bias.as_ref(), relu);
                    let mut plan = cuda::Conv2d::new(
                        &gpu,
                        s,
                        &weights,
                        bias.as_ref(),
                        if relu {
                            Activation::Relu
                        } else {
                            Activation::Identity
                        },
                        64 << 20,
                    )
                    .unwrap();
                    assert!(plan.workspace_bytes() <= 64 << 20);
                    if let Some(info) = plan.cudnn_plan_info() {
                        assert!(info.engine_index >= 0);
                        assert!(info.workspace_bytes <= plan.workspace_bytes());
                        if policy == cuda::ConvolutionPolicy::RequireDeterministic {
                            assert!(info.deterministic);
                        }
                    } else {
                        assert!(plan.implementation().starts_with("cuBLAS"));
                    }
                    if s.transposed
                        && s.output_padding
                            .iter()
                            .zip(s.stride)
                            .any(|(&p, stride)| p >= stride)
                    {
                        assert!(
                            cuda::Conv2d::new(
                                &gpu,
                                s,
                                &weights,
                                bias.as_ref(),
                                Activation::Identity,
                                0,
                            )
                            .is_err()
                        );
                    }
                    weights.data_mut().fill(-100.0);
                    if let Some(b) = &mut bias {
                        b.data_mut().fill(77.0);
                    }
                    let input = gpu.upload(&x).unwrap();
                    let mut out = gpu.zeros(s.output_shape().unwrap()).unwrap();
                    plan.run(&input, &mut out).unwrap();
                    let first = gpu.download(&out).unwrap();
                    close(first.data(), &expected, 5e-6, 3e-5);
                    no_allocations(|| plan.run(&input, &mut out).unwrap());
                    if policy == cuda::ConvolutionPolicy::RequireDeterministic {
                        assert!(
                            first.data().iter().map(|x| x.to_bits()).eq(gpu
                                .download(&out)
                                .unwrap()
                                .data()
                                .iter()
                                .map(|x| x.to_bits()))
                        );
                    }
                    let input = gpu.upload(&x).unwrap();
                    let mut out = gpu.zeros(s.output_shape().unwrap()).unwrap();
                    plan.run(&input, &mut out).unwrap();
                    close(gpu.download(&out).unwrap().data(), &expected, 5e-6, 3e-5);
                    let other = cuda::Cuda::new(0).unwrap();
                    let foreign = other.upload(&x).unwrap();
                    assert!(plan.run(&foreign, &mut out).is_err());
                }
            }
        }
    }
}

#[cfg(feature = "cudnn")]
#[test]
fn cudnn_zero_workspace_never_relaxes_policy_or_returns_an_unbounded_plan() {
    use fast_audio::{Error, cuda};
    for policy in [
        cuda::ConvolutionPolicy::AllowNonDeterministic,
        cuda::ConvolutionPolicy::RequireDeterministic,
    ] {
        let gpu = cuda::Cuda::with_convolution_policy(0, policy).unwrap();
        for s in cases().into_iter().take(5) {
            match cuda::Conv2d::new(&gpu, s, &data(s.weight_shape()), None, Activation::Relu, 0) {
                Ok(plan) => {
                    assert_eq!(plan.workspace_bytes(), 0);
                    if policy == cuda::ConvolutionPolicy::RequireDeterministic {
                        assert!(plan.cudnn_plan_info().unwrap().deterministic);
                    }
                }
                Err(Error::Unsupported(message)) => {
                    assert!(message.contains(policy.name()));
                    assert!(message.contains("workspace <= 0 bytes"));
                    println!("unsupported {policy:?} {s:?}");
                }
                Err(other) => panic!("unexpected error for valid geometry: {other}"),
            }
        }
    }
}

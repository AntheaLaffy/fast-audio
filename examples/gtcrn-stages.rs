//! Replay the pinned model's actual spectral frontend and all 2D convolution layers.
use fast_audio::{
    Activation, Conv2dSpec, StftSpec, Tensor,
    cpu::{self, Cpu},
};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{collections::HashMap, fs, path::Path, time::Instant};
mod support;
use support::{AnyResult, Blob, load, save};

#[derive(Deserialize)]
struct Layer {
    name: String,
    #[serde(rename = "type")]
    kind: String,
    kernel: Option<[usize; 2]>,
    stride: Option<[usize; 2]>,
    padding: Option<[usize; 2]>,
    dilation: Option<[usize; 2]>,
    groups: Option<usize>,
    transposed: Option<bool>,
    output_padding: Option<[usize; 2]>,
    input: Option<Blob>,
    output: Option<Blob>,
    parameters: Option<HashMap<String, Blob>>,
}
#[derive(Deserialize)]
struct Case {
    name: String,
    batch: usize,
    input_len: usize,
    tensors: HashMap<String, Blob>,
    leaf_operators: Vec<Layer>,
}
#[derive(Deserialize)]
struct Reference {
    cases: Vec<Case>,
}
enum Backend {
    Cpu(Cpu),
    #[cfg(feature = "cudnn")]
    Cuda(fast_audio::cuda::Cuda),
}

fn spectral(
    backend: &Backend,
    s: StftSpec,
    x: &Tensor,
    window: &Tensor,
    enhanced: &Tensor,
) -> AnyResult<(Tensor, Tensor)> {
    let spec = s.centered()?;
    let mut frequency = Tensor::zeros(spec.output_shape())?;
    // Model reference layout is [B,F,T,2]; public native spectra are [B,T,F,2].
    for b in 0..s.batch {
        for f in 0..s.bins() {
            for t in 0..spec.frames() {
                let source = ((b * s.bins() + f) * spec.frames() + t) * 2;
                let target = ((b * spec.frames() + t) * s.bins() + f) * 2;
                frequency.data_mut()[target..target + 2]
                    .copy_from_slice(&enhanced.data()[source..source + 2]);
            }
        }
    }
    match backend {
        Backend::Cpu(cpu) => {
            let mut forward = cpu::CenteredStft::with_window(*cpu, spec, window)?;
            let mut inverse = cpu::Istft::with_window(*cpu, spec, window)?;
            let mut spectrum = Tensor::zeros(spec.output_shape())?;
            let mut waveform = Tensor::zeros(spec.input_shape())?;
            forward.run(x, &mut spectrum)?;
            inverse.run(&frequency, &mut waveform)?;
            Ok((spectrum, waveform))
        }
        #[cfg(feature = "cudnn")]
        Backend::Cuda(gpu) => {
            use fast_audio::cuda;
            let mut forward = cuda::CenteredStft::with_window(gpu, spec, window)?;
            let mut inverse = cuda::Istft::with_window(gpu, spec, window)?;
            let x = gpu.upload(x)?;
            let frequency = gpu.upload(&frequency)?;
            let mut spectrum = gpu.zeros(spec.output_shape())?;
            let mut waveform = gpu.zeros(spec.input_shape())?;
            forward.run(&x, &mut spectrum)?;
            inverse.run(&frequency, &mut waveform)?;
            Ok((gpu.download(&spectrum)?, gpu.download(&waveform)?))
        }
    }
}
fn convolution(
    backend: &Backend,
    s: Conv2dSpec,
    x: &Tensor,
    w: &Tensor,
    b: Option<&Tensor>,
) -> AnyResult<(Tensor, Value)> {
    match backend {
        Backend::Cpu(cpu) => {
            let begin = Instant::now();
            let mut plan = cpu::Conv2d::new(*cpu, s, w, b, Activation::Identity)?;
            let setup = begin.elapsed().as_secs_f64() * 1e6;
            let mut y = Tensor::zeros(s.output_shape()?)?;
            for _ in 0..20 {
                plan.run(x, &mut y)?;
            }
            let mut samples = Vec::new();
            for _ in 0..7 {
                let begin = Instant::now();
                for _ in 0..50 {
                    plan.run(x, &mut y)?;
                }
                samples.push(begin.elapsed().as_secs_f64() * 1e6 / 50.0);
            }
            Ok((
                y,
                json!({"wall_us":samples,"setup_us":setup,"workspace_bytes":plan.workspace_bytes(),"parameter_bytes":plan.parameter_bytes(),"implementation":plan.implementation()}),
            ))
        }
        #[cfg(feature = "cudnn")]
        Backend::Cuda(gpu) => {
            let x = gpu.upload(x)?;
            let begin = Instant::now();
            let mut plan =
                fast_audio::cuda::Conv2d::new(gpu, s, w, b, Activation::Identity, 64 << 20)?;
            let setup = begin.elapsed().as_secs_f64() * 1e6;
            let mut y = gpu.zeros(s.output_shape()?)?;
            for _ in 0..20 {
                plan.run(&x, &mut y)?;
            }
            gpu.synchronize()?;
            let mut wall = Vec::new();
            let mut device = Vec::new();
            for _ in 0..7 {
                let sample = gpu.measure(50, || plan.run(&x, &mut y))?;
                wall.push(sample.wall_us);
                device.push(sample.device_us);
            }
            Ok((
                gpu.download(&y)?,
                json!({"wall_us":wall,"device_us":device,"setup_us":setup,"workspace_bytes":plan.workspace_bytes(),"parameter_bytes":plan.parameter_bytes(),"implementation":plan.implementation()}),
            ))
        }
    }
}
fn main() -> AnyResult<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 3 {
        return Err(
            "usage: gtcrn-stages reference|mkl|cudnn REFERENCE_DIRECTORY OUTPUT_DIRECTORY".into(),
        );
    }
    let backend = match args[0].as_str() {
        "reference" => Backend::Cpu(Cpu::Reference),
        #[cfg(feature = "mkl")]
        "mkl" => Backend::Cpu(Cpu::mkl(1)?),
        #[cfg(feature = "cudnn")]
        "cudnn" => Backend::Cuda(fast_audio::cuda::Cuda::new(0)?),
        _ => return Err("unknown or unavailable backend".into()),
    };
    let reference = Path::new(&args[1]);
    let out = Path::new(&args[2]);
    if out.exists() {
        return Err("use a new output directory".into());
    }
    fs::create_dir_all(out)?;
    let data: Reference = serde_json::from_slice(&fs::read(reference.join("reference.json"))?)?;
    let mut audio = Vec::new();
    let mut layers = Vec::new();
    for case in data.cases {
        let folder = reference.join(&case.name);
        let s = StftSpec {
            batch: case.batch,
            input_len: case.input_len,
            fft_size: 512,
            hop: 256,
        };
        let x = load(
            &folder,
            case.tensors.get("input").ok_or("missing waveform")?,
        )?;
        let w = load(&folder, case.tensors.get("window").ok_or("missing window")?)?;
        let enhanced = load(
            &folder,
            case.tensors
                .get("enhanced")
                .ok_or("missing enhanced spectrum")?,
        )?;
        let (spectrum, waveform) = spectral(&backend, s, &x, &w, &enhanced)?;
        let spectrum_file = format!("{}-spectrum.f32", case.name);
        let waveform_file = format!("{}-waveform.f32", case.name);
        save(&out.join(&spectrum_file), &spectrum)?;
        save(&out.join(&waveform_file), &waveform)?;
        audio.push(json!({"case":case.name,"spectrum":{"file":spectrum_file,"shape":spectrum.shape().dims()},"waveform":{"file":waveform_file,"shape":waveform.shape().dims()}}));
        for layer in case
            .leaf_operators
            .into_iter()
            .filter(|l| l.kind == "Conv2d" || l.kind == "ConvTranspose2d")
        {
            let x = load(
                &folder,
                layer
                    .input
                    .as_ref()
                    .ok_or("missing captured convolution input")?,
            )?;
            let p = layer
                .parameters
                .as_ref()
                .ok_or("missing convolution parameters")?;
            let w = load(&folder, p.get("weight").ok_or("missing weight")?)?;
            let b = p.get("bias").map(|v| load(&folder, v)).transpose()?;
            let &[batch, in_channels, ih, iw] = x.shape().dims() else {
                return Err("expected NCHW input".into());
            };
            let expected = layer
                .output
                .as_ref()
                .ok_or("missing captured convolution output")?;
            let spec = Conv2dSpec {
                batch,
                in_channels,
                out_channels: expected.shape[1],
                input: [ih, iw],
                kernel: layer.kernel.ok_or("missing kernel")?,
                stride: layer.stride.ok_or("missing stride")?,
                padding: layer.padding.ok_or("missing padding")?,
                dilation: layer.dilation.ok_or("missing dilation")?,
                groups: layer.groups.ok_or("missing groups")?,
                transposed: layer.transposed.ok_or("missing direction")?,
                output_padding: layer.output_padding.ok_or("missing output padding")?,
            };
            if spec.output_shape()?.as_slice() != expected.shape {
                return Err("captured convolution geometry mismatch".into());
            }
            let (y, timing) = convolution(&backend, spec, &x, &w, b.as_ref())?;
            let file = format!("{}-{}.f32", case.name, layer.name.replace('.', "-"));
            save(&out.join(&file), &y)?;
            layers.push(json!({"case":case.name,"layer":layer.name,"output":{"file":file,"shape":y.shape().dims()},"timing":timing}));
        }
    }
    fs::write(
        out.join("stages.json"),
        serde_json::to_string_pretty(
            &json!({"schema_version":1,"backend":args[0],"audio":audio,"layers":layers}),
        )? + "\n",
    )?;
    Ok(())
}

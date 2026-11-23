//! Complete native waveform -> STFT -> GTCRN -> inverse path from exported FP32 weights.
use fast_audio::{
    StftSpec, Tensor,
    cpu::{self, Cpu},
    models::gtcrn::{self, Parameters},
};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{collections::BTreeMap, fs, path::Path, time::Instant};
mod support;
use support::{AnyResult, Blob, load, save};

#[derive(Deserialize)]
struct Case {
    name: String,
    batch: usize,
    input_len: usize,
    tensors: BTreeMap<String, Blob>,
}
#[derive(Deserialize)]
struct Reference {
    cases: Vec<Case>,
}
#[derive(Deserialize)]
struct Bundle {
    format: String,
    parameters: BTreeMap<String, Blob>,
}
#[derive(Clone, Copy)]
struct Counts {
    warmup: usize,
    samples: usize,
    iterations: usize,
    traces: bool,
    pipeline_only: bool,
}
fn timed(counts: Counts, mut f: impl FnMut() -> AnyResult<()>) -> AnyResult<Value> {
    for _ in 0..counts.warmup {
        f()?;
    }
    let mut wall = Vec::new();
    for _ in 0..counts.samples {
        let begin = Instant::now();
        for _ in 0..counts.iterations {
            f()?;
        }
        wall.push(begin.elapsed().as_secs_f64() * 1e6 / counts.iterations as f64);
    }
    Ok(json!({"wall_us":wall}))
}
fn save_blob(folder: &Path, file: String, value: &Tensor) -> AnyResult<Value> {
    save(&folder.join(&file), value)?;
    Ok(json!({"file":file,"shape":value.shape().dims()}))
}
fn cpu_case(
    cpu: Cpu,
    case: &Case,
    x: &Tensor,
    window: &Tensor,
    parameters: &Parameters,
    out: &Path,
    counts: Counts,
) -> AnyResult<Value> {
    let spec = StftSpec {
        batch: case.batch,
        input_len: case.input_len,
        fft_size: 512,
        hop: 256,
    }
    .centered()?;
    let begin = Instant::now();
    let model = gtcrn::build(case.batch, spec.frames(), parameters)?;
    let mut graph = cpu::Graph::new(cpu, model.program)?;
    let mut stft = cpu::CenteredStft::with_window(cpu, spec, window)?;
    let mut inverse = cpu::Istft::with_window(cpu, spec, window)?;
    let mut spectrum = Tensor::zeros(spec.output_shape())?;
    let mut enhanced = Tensor::zeros(spec.output_shape())?;
    let mut waveform = Tensor::zeros(spec.input_shape())?;
    let setup_us = begin.elapsed().as_secs_f64() * 1e6;
    stft.run(x, &mut spectrum)?;
    graph.run(&spectrum, &mut enhanced)?;
    inverse.run(&enhanced, &mut waveform)?;
    let network = if counts.pipeline_only {
        Value::Null
    } else {
        timed(counts, || {
            graph.run(&spectrum, &mut enhanced)?;
            Ok(())
        })?
    };
    let pipeline = timed(counts, || {
        stft.run(x, &mut spectrum)?;
        graph.run(&spectrum, &mut enhanced)?;
        inverse.run(&enhanced, &mut waveform)?;
        Ok(())
    })?;
    let mut stages = BTreeMap::new();
    for (name, value) in model.probes.into_iter().filter(|_| counts.traces) {
        let t = graph
            .intermediate(value)
            .ok_or("missing model intermediate")?;
        stages.insert(
            name.clone(),
            save_blob(
                out,
                format!("{}-{}.f32", case.name, name.replace('.', "-")),
                t,
            )?,
        );
    }
    Ok(
        json!({"case":case.name,"spectrum":save_blob(out,format!("{}-spectrum.f32",case.name),&spectrum)?,
        "enhanced":save_blob(out,format!("{}-enhanced.f32",case.name),&enhanced)?,
        "waveform":save_blob(out,format!("{}-waveform.f32",case.name),&waveform)?,"model_stages":stages,
        "setup_us":setup_us,"graph_workspace_bytes":graph.workspace_bytes(),"parameter_bytes":graph.parameter_bytes(),
        "inverse_workspace_bytes":inverse.workspace_bytes(),"timing":{"model":network,"pipeline":pipeline}}),
    )
}
#[cfg(feature = "cudnn")]
fn cuda_case(
    gpu: &fast_audio::cuda::Cuda,
    case: &Case,
    x: &Tensor,
    window: &Tensor,
    parameters: &Parameters,
    out: &Path,
    counts: Counts,
) -> AnyResult<Value> {
    use fast_audio::cuda;
    let spec = StftSpec {
        batch: case.batch,
        input_len: case.input_len,
        fft_size: 512,
        hop: 256,
    }
    .centered()?;
    let x = gpu.upload(x)?;
    let begin = Instant::now();
    let model = gtcrn::build(case.batch, spec.frames(), parameters)?;
    let mut graph = cuda::Graph::new(gpu, model.program)?;
    let mut stft = cuda::CenteredStft::with_window(gpu, spec, window)?;
    let mut inverse = cuda::Istft::with_window(gpu, spec, window)?;
    let mut spectrum = gpu.zeros(spec.output_shape())?;
    let mut enhanced = gpu.zeros(spec.output_shape())?;
    let mut waveform = gpu.zeros(spec.input_shape())?;
    gpu.synchronize()?;
    let setup_us = begin.elapsed().as_secs_f64() * 1e6;
    stft.run(&x, &mut spectrum)?;
    graph.run(&spectrum, &mut enhanced)?;
    inverse.run(&enhanced, &mut waveform)?;
    gpu.synchronize()?;
    let mut measure = |pipeline: bool| -> AnyResult<Value> {
        let mut f = || {
            if pipeline {
                stft.run(&x, &mut spectrum)?;
            }
            graph.run(&spectrum, &mut enhanced)?;
            if pipeline {
                inverse.run(&enhanced, &mut waveform)?;
            }
            Ok(())
        };
        for _ in 0..counts.warmup {
            f()?;
        }
        gpu.synchronize()?;
        let mut wall = Vec::new();
        let mut device = Vec::new();
        for _ in 0..counts.samples {
            let sample = gpu.measure(counts.iterations, &mut f)?;
            wall.push(sample.wall_us);
            device.push(sample.device_us);
        }
        Ok(json!({"wall_us":wall,"device_us":device}))
    };
    let network = if counts.pipeline_only {
        Value::Null
    } else {
        measure(false)?
    };
    let pipeline = measure(true)?;
    let mut stages = BTreeMap::new();
    for (name, value) in model.probes.into_iter().filter(|_| counts.traces) {
        let tensor = gpu.download(
            graph
                .intermediate(value)
                .ok_or("missing model intermediate")?,
        )?;
        stages.insert(
            name.clone(),
            save_blob(
                out,
                format!("{}-{}.f32", case.name, name.replace('.', "-")),
                &tensor,
            )?,
        );
    }
    let convolution_plans: Vec<_> = graph.cudnn_convolution_plans().map(|(step, info)| {
        json!({"step":step,"engine_index":info.engine_index,"workspace_bytes":info.workspace_bytes,
            "numerical_notes":info.numerical_notes,"deterministic":info.deterministic,
            "fused_epilogue":info.fused_epilogue})
    }).collect();
    Ok(
        json!({"case":case.name,"spectrum":save_blob(out,format!("{}-spectrum.f32",case.name),&gpu.download(&spectrum)?)?,
        "enhanced":save_blob(out,format!("{}-enhanced.f32",case.name),&gpu.download(&enhanced)?)?,
        "waveform":save_blob(out,format!("{}-waveform.f32",case.name),&gpu.download(&waveform)?)?,"model_stages":stages,
        "setup_us":setup_us,"graph_workspace_bytes":graph.workspace_bytes(),"parameter_bytes":graph.parameter_bytes(),
        "inverse_workspace_bytes":inverse.workspace_bytes(),"convolution_plans":convolution_plans,
        "timing":{"model":network,"pipeline":pipeline}}),
    )
}
fn main() -> AnyResult<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() < 3 {
        return Err(
            "usage: gtcrn-infer reference|mkl|cudnn REFERENCE OUT [WARMUP SAMPLES ITERATIONS] [--case NAME] [--no-trace] [--pipeline-only] [--deterministic-convolution]"
                .into(),
        );
    }
    let numbered = args.get(3).is_some_and(|arg| !arg.starts_with("--"));
    if numbered && args.len() < 6 {
        return Err("provide all three timing counts".into());
    }
    let mut counts = if numbered && args.len() >= 6 {
        Counts {
            warmup: args[3].parse()?,
            samples: args[4].parse()?,
            iterations: args[5].parse()?,
            traces: true,
            pipeline_only: false,
        }
    } else {
        Counts {
            warmup: 2,
            samples: 3,
            iterations: 5,
            traces: true,
            pipeline_only: false,
        }
    };
    let mut selected = None;
    let mut deterministic_convolution = false;
    let mut options = args[if numbered { 6 } else { 3 }..].iter();
    while let Some(option) = options.next() {
        match option.as_str() {
            "--case" => selected = Some(options.next().ok_or("--case needs a name")?.as_str()),
            "--no-trace" => counts.traces = false,
            "--pipeline-only" => counts.pipeline_only = true,
            "--deterministic-convolution" => deterministic_convolution = true,
            _ => return Err(format!("unknown option {option}").into()),
        }
    }
    if deterministic_convolution && (args[0] != "cudnn" || !cfg!(feature = "cudnn")) {
        return Err("--deterministic-convolution requires the cuDNN backend".into());
    }
    if counts.samples == 0 || counts.iterations == 0 {
        return Err("positive sample and iteration counts required".into());
    }
    let reference = Path::new(&args[1]);
    let out = Path::new(&args[2]);
    if out.exists() {
        return Err("use a new output directory".into());
    }
    let bundle: Bundle = serde_json::from_slice(&fs::read(reference.join("parameters.json"))?)?;
    if bundle.format != "fast-audio-gtcrn-f32-v1" {
        return Err("unknown parameter bundle format".into());
    }
    let parameters: Parameters = bundle
        .parameters
        .into_iter()
        .map(|(name, blob)| Ok((name, load(reference, &blob)?)))
        .collect::<AnyResult<_>>()?;
    let data: Reference = serde_json::from_slice(&fs::read(reference.join("reference.json"))?)?;
    if selected.is_some_and(|name| !data.cases.iter().any(|case| case.name == name)) {
        return Err("unknown requested case".into());
    }
    #[cfg(feature = "cudnn")]
    let gpu = if args[0] == "cudnn" {
        use fast_audio::cuda::{ConvolutionPolicy, Cuda};
        Some(Cuda::with_convolution_policy(
            0,
            if deterministic_convolution {
                ConvolutionPolicy::RequireDeterministic
            } else {
                ConvolutionPolicy::AllowNonDeterministic
            },
        )?)
    } else {
        None
    };
    fs::create_dir_all(out)?;
    let mut cases = Vec::new();
    for case in data
        .cases
        .iter()
        .filter(|case| selected.is_none_or(|name| case.name == name))
    {
        let folder = reference.join(&case.name);
        let x = load(&folder, case.tensors.get("input").ok_or("missing input")?)?;
        let window = load(&folder, case.tensors.get("window").ok_or("missing window")?)?;
        let record = match args[0].as_str() {
            "reference" => cpu_case(Cpu::Reference, case, &x, &window, &parameters, out, counts)?,
            #[cfg(feature = "mkl")]
            "mkl" => cpu_case(Cpu::mkl(1)?, case, &x, &window, &parameters, out, counts)?,
            #[cfg(feature = "cudnn")]
            "cudnn" => cuda_case(
                gpu.as_ref().unwrap(),
                case,
                &x,
                &window,
                &parameters,
                out,
                counts,
            )?,
            _ => return Err("unknown or unavailable backend".into()),
        };
        println!("{} complete", case.name);
        cases.push(record);
    }
    fs::write(
        out.join("inference.json"),
        serde_json::to_string_pretty(&json!({"schema_version":1,"backend":args[0],
        "compiled_features":{"mkl":cfg!(feature="mkl"),"onednn":cfg!(feature="onednn"),"cuda":cfg!(feature="cuda"),"cudnn":cfg!(feature="cudnn")},
        "protocol":{"warmup":counts.warmup,"samples":counts.samples,"iterations":counts.iterations,"traces":counts.traces,"pipeline_only":counts.pipeline_only,
            "convolution_policy":if args[0]=="cudnn" {Some(if deterministic_convolution {"require_deterministic"} else {"allow_nondeterministic"})} else {None},
            "scope":"complete native offline pipeline; native-only exploratory timings"},"cases":cases}))?
            + "\n",
    )?;
    Ok(())
}

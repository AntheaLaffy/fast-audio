use fast_audio::{
    Activation, Conv1dSpec, StftSpec, Tensor,
    cpu::{self, Cpu},
};
use serde::{Deserialize, Serialize};
use std::{error::Error, fs, path::PathBuf, time::Instant};

type AnyResult<T> = Result<T, Box<dyn Error>>;
#[derive(Clone, Deserialize, Serialize)]
struct Suite {
    schema_version: u32,
    cases: Vec<Case>,
}
#[derive(Clone, Deserialize, Serialize)]
struct Case {
    name: String,
    #[serde(flatten)]
    workload: Workload,
}
#[derive(Clone, Deserialize, Serialize)]
#[serde(tag = "op", rename_all = "snake_case")]
enum Workload {
    Matmul {
        m: usize,
        n: usize,
        k: usize,
    },
    Conv1d {
        batch: usize,
        in_channels: usize,
        out_channels: usize,
        input_len: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
        bias: bool,
        relu: bool,
    },
    Stft {
        batch: usize,
        input_len: usize,
        fft_size: usize,
        hop: usize,
    },
}
impl Workload {
    fn conv(&self) -> (Conv1dSpec, Activation) {
        let Self::Conv1d {
            batch,
            in_channels,
            out_channels,
            input_len,
            kernel,
            stride,
            padding,
            dilation,
            groups,
            relu,
            ..
        } = *self
        else {
            unreachable!()
        };
        (
            Conv1dSpec {
                batch,
                in_channels,
                out_channels,
                input_len,
                kernel,
                stride,
                padding,
                dilation,
                groups,
            },
            if relu {
                Activation::Relu
            } else {
                Activation::Identity
            },
        )
    }
    fn stft(&self) -> StftSpec {
        let Self::Stft {
            batch,
            input_len,
            fft_size,
            hop,
        } = *self
        else {
            unreachable!()
        };
        StftSpec {
            batch,
            input_len,
            fft_size,
            hop,
        }
    }
}

// Fixed integer hash shared with tools/compare_torch.py. Avoid implementation-specific RNGs.
fn data(dims: impl AsRef<[usize]>, salt: u32) -> fast_audio::Result<Tensor> {
    let mut tensor = Tensor::zeros(dims)?;
    for (i, value) in tensor.data_mut().iter_mut().enumerate() {
        let state = (i as u32).wrapping_mul(747796405).wrapping_add(salt);
        let word = ((state >> ((state >> 28) + 4)) ^ state).wrapping_mul(277803737);
        let bits = ((word >> 22) ^ word) & 65535;
        *value = (bits as i32 - 32768) as f32 / 32768.0;
    }
    Ok(tensor)
}
struct Inputs {
    x: Tensor,
    w: Option<Tensor>,
    bias: Option<Tensor>,
    y: Tensor,
}
fn inputs(work: &Workload) -> fast_audio::Result<Inputs> {
    Ok(match *work {
        Workload::Matmul { m, n, k } => Inputs {
            x: data([m, k], 11)?,
            w: Some(data([k, n], 29)?),
            bias: None,
            y: Tensor::zeros([m, n])?,
        },
        Workload::Conv1d { bias, .. } => {
            let (spec, _) = work.conv();
            let out = spec.output_shape()?;
            Inputs {
                x: data(spec.input_shape(), 11)?,
                w: Some(data(spec.weight_shape(), 29)?),
                bias: if bias {
                    Some(data([spec.out_channels], 47)?)
                } else {
                    None
                },
                y: Tensor::zeros(out)?,
            }
        }
        Workload::Stft { .. } => {
            let spec = work.stft();
            Inputs {
                x: data(spec.input_shape(), 11)?,
                w: None,
                bias: None,
                y: Tensor::zeros(spec.output_shape()?)?,
            }
        }
    })
}

struct Options {
    backend: String,
    case: String,
    threads: u16,
    iterations: usize,
    warmup: usize,
    samples: usize,
    output: Option<PathBuf>,
    suite: Suite,
}
impl Options {
    fn parse() -> AnyResult<Self> {
        let mut opts = Self {
            backend: "mkl".into(),
            case: "gemm-stream".into(),
            threads: 1,
            iterations: 50,
            warmup: 20,
            samples: 9,
            output: None,
            suite: serde_json::from_str(include_str!("../../benchmarks/workloads.json"))?,
        };
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            if arg == "--list" {
                println!("{}", serde_json::to_string_pretty(&opts.suite)?);
                std::process::exit(0);
            }
            if arg == "--help" {
                println!(
                    "fa-bench --backend reference|mkl|onednn|intel|cuda|cudnn|nvidia --case NAME [--threads N] [--iterations N] [--samples N] [--warmup N] [--output FILE] [--workloads JSON]"
                );
                std::process::exit(0);
            }
            let value = args.next().ok_or("every option needs a value")?;
            match arg.as_str() {
                "--backend" => opts.backend = value,
                "--case" => opts.case = value,
                "--threads" => opts.threads = value.parse()?,
                "--iterations" => opts.iterations = value.parse()?,
                "--warmup" => opts.warmup = value.parse()?,
                "--samples" => opts.samples = value.parse()?,
                "--output" => opts.output = Some(value.into()),
                "--workloads" => opts.suite = serde_json::from_str(&fs::read_to_string(value)?)?,
                _ => return Err(format!("unknown option {arg}").into()),
            }
        }
        if opts.iterations == 0 || opts.samples == 0 || opts.threads == 0 {
            return Err("iterations, samples, and threads must be positive".into());
        }
        if opts.suite.schema_version != 1 {
            return Err("unsupported workload schema".into());
        }
        Ok(opts)
    }
}
#[derive(Serialize)]
struct Sample {
    wall_us: f64,
    device_us: Option<f64>,
}
#[derive(Serialize)]
struct Report {
    schema_version: u32,
    case: Case,
    backend: String,
    device: String,
    threads: u16,
    iterations: usize,
    warmup: usize,
    setup_us: f64,
    samples: Vec<Sample>,
    output_shape: Vec<usize>,
    output_sum_f64: f64,
    workspace_bytes: Option<usize>,
    implementation: Option<String>,
    parameter_bytes: Option<usize>,
}

enum CpuOp {
    Matmul(Cpu),
    Conv(cpu::Conv1d),
    Stft(cpu::Stft),
    #[cfg(feature = "onednn")]
    OneDnn(fast_audio::onednn::Conv1d),
    Prepared(cpu::PreparedConv1d),
}
impl CpuOp {
    fn run(&mut self, args: &mut Inputs) -> fast_audio::Result<()> {
        match self {
            Self::Matmul(cpu) => cpu.matmul(&args.x, args.w.as_ref().unwrap(), &mut args.y),
            Self::Conv(op) => op.run(
                &args.x,
                args.w.as_ref().unwrap(),
                args.bias.as_ref(),
                &mut args.y,
            ),
            Self::Stft(op) => op.run(&args.x, &mut args.y),
            #[cfg(feature = "onednn")]
            Self::OneDnn(op) => op.run(&args.x, &mut args.y),
            Self::Prepared(op) => op.run(&args.x, &mut args.y),
        }
    }
}
fn run_cpu(case: &Case, opts: &Options) -> AnyResult<(Tensor, Report)> {
    if (opts.backend == "onednn" || opts.backend == "intel") && !cfg!(feature = "onednn") {
        return Err("compile with --features mkl,onednn".into());
    }
    let start = Instant::now();
    let cpu = match opts.backend.as_str() {
        "reference" => Cpu::Reference,
        #[cfg(feature = "mkl")]
        "mkl" | "onednn" | "intel" => Cpu::mkl(opts.threads)?,
        _ => return Err("CPU backend unavailable; compile with --features mkl".into()),
    };
    let mut inputs = inputs(&case.workload)?;
    let mut op = match case.workload {
        Workload::Matmul { .. } => CpuOp::Matmul(cpu),
        Workload::Conv1d { .. } => {
            let (spec, act) = case.workload.conv();
            if opts.backend == "intel" {
                CpuOp::Prepared(cpu::PreparedConv1d::new(
                    cpu,
                    spec,
                    inputs.w.as_ref().unwrap(),
                    inputs.bias.as_ref(),
                    act,
                )?)
            } else if opts.backend == "onednn" {
                #[cfg(feature = "onednn")]
                {
                    CpuOp::OneDnn(fast_audio::onednn::Conv1d::new(
                        spec,
                        inputs.w.as_ref().unwrap(),
                        inputs.bias.as_ref(),
                        act,
                        opts.threads,
                    )?)
                }
                #[cfg(not(feature = "onednn"))]
                {
                    return Err("compile with --features mkl,onednn".into());
                }
            } else {
                CpuOp::Conv(cpu::Conv1d::new(cpu, spec, act)?)
            }
        }
        Workload::Stft { .. } => CpuOp::Stft(cpu::Stft::new(cpu, case.workload.stft())?),
    };
    let setup_us = start.elapsed().as_secs_f64() * 1e6;
    let workspace = match &op {
        CpuOp::Conv(p) => Some(p.workspace_bytes()),
        #[cfg(feature = "onednn")]
        CpuOp::OneDnn(p) => Some(p.workspace_bytes()),
        CpuOp::Prepared(p) => Some(p.workspace_bytes()),
        _ => None,
    };
    for _ in 0..opts.warmup {
        op.run(&mut inputs)?;
    }
    let mut samples = Vec::new();
    for _ in 0..opts.samples {
        let time = Instant::now();
        for _ in 0..opts.iterations {
            op.run(&mut inputs)?;
        }
        samples.push(Sample {
            wall_us: time.elapsed().as_secs_f64() * 1e6 / opts.iterations as f64,
            device_us: None,
        });
    }
    #[allow(unused_mut)]
    let mut report = report(
        case,
        opts,
        setup_us,
        samples,
        &inputs.y,
        "CPU".into(),
        workspace,
    );
    #[cfg(feature = "onednn")]
    if let CpuOp::OneDnn(plan) = &op {
        report.implementation = Some(plan.implementation().into());
        report.parameter_bytes = Some(plan.parameter_bytes());
    }
    if let CpuOp::Prepared(plan) = &op {
        report.implementation = Some(plan.implementation().into());
        report.parameter_bytes = Some(plan.parameter_bytes());
    }
    Ok((inputs.y, report))
}

#[cfg(feature = "cuda")]
fn run_cuda(case: &Case, opts: &Options) -> AnyResult<(Tensor, Report)> {
    use fast_audio::cuda::{self, Cuda};
    let start = Instant::now();
    if (opts.backend == "cudnn" || opts.backend == "nvidia") && !cfg!(feature = "cudnn") {
        return Err("compile with --features cudnn".into());
    }
    let host = inputs(&case.workload)?;
    let gpu = Cuda::new(0)?;
    let x = gpu.upload(&host.x)?;
    let w = host.w.as_ref().map(|w| gpu.upload(w)).transpose()?;
    let bias = host.bias.as_ref().map(|b| gpu.upload(b)).transpose()?;
    let mut y = gpu.zeros(host.y.shape().dims())?;
    enum Op {
        Matmul,
        Conv(cuda::Conv1d),
        Stft(cuda::Stft),
    }
    let mut op = match case.workload {
        Workload::Matmul { .. } => Op::Matmul,
        Workload::Conv1d { .. } => {
            let (spec, act) = case.workload.conv();
            if opts.backend == "nvidia" {
                #[cfg(feature = "cudnn")]
                {
                    Op::Conv(cuda::Conv1d::planned(
                        &gpu,
                        spec,
                        host.bias.is_some(),
                        act,
                        64 * 1024 * 1024,
                    )?)
                }
                #[cfg(not(feature = "cudnn"))]
                {
                    return Err("compile with --features cudnn".into());
                }
            } else if opts.backend == "cudnn" {
                #[cfg(feature = "cudnn")]
                {
                    Op::Conv(cuda::Conv1d::with_cudnn(
                        &gpu,
                        spec,
                        host.bias.is_some(),
                        act,
                        64 * 1024 * 1024,
                    )?)
                }
                #[cfg(not(feature = "cudnn"))]
                {
                    return Err("compile with --features cudnn".into());
                }
            } else {
                Op::Conv(cuda::Conv1d::new(&gpu, spec, act)?)
            }
        }
        Workload::Stft { .. } => Op::Stft(cuda::Stft::new(&gpu, case.workload.stft())?),
    };
    gpu.synchronize()?;
    let setup_us = start.elapsed().as_secs_f64() * 1e6;
    let workspace = match &op {
        Op::Conv(p) => Some(p.workspace_bytes()),
        _ => None,
    };
    let implementation = match &op {
        Op::Conv(p) => Some(p.implementation().to_owned()),
        _ => None,
    };
    let mut run = || match &mut op {
        Op::Matmul => gpu.matmul(&x, w.as_ref().unwrap(), &mut y),
        Op::Conv(plan) => plan.run(&x, w.as_ref().unwrap(), bias.as_ref(), &mut y),
        Op::Stft(plan) => plan.run(&x, &mut y),
    };
    for _ in 0..opts.warmup {
        run()?;
    }
    let mut samples = Vec::new();
    for _ in 0..opts.samples {
        let timing = gpu.measure(opts.iterations, &mut run)?;
        samples.push(Sample {
            wall_us: timing.wall_us,
            device_us: Some(timing.device_us),
        });
    }
    let output = gpu.download(&y)?;
    let mut report = report(
        case,
        opts,
        setup_us,
        samples,
        &output,
        gpu.device_name()?,
        workspace,
    );
    report.implementation = implementation;
    Ok((output, report))
}

fn report(
    case: &Case,
    opts: &Options,
    setup_us: f64,
    samples: Vec<Sample>,
    output: &Tensor,
    device: String,
    workspace_bytes: Option<usize>,
) -> Report {
    Report {
        schema_version: 1,
        case: case.clone(),
        backend: opts.backend.clone(),
        device,
        threads: opts.threads,
        iterations: opts.iterations,
        warmup: opts.warmup,
        setup_us,
        samples,
        output_shape: output.shape().dims().into(),
        output_sum_f64: output.data().iter().map(|&x| x as f64).sum(),
        workspace_bytes,
        implementation: None,
        parameter_bytes: None,
    }
}

fn main() -> AnyResult<()> {
    let opts = Options::parse()?;
    let case = opts
        .suite
        .cases
        .iter()
        .find(|case| case.name == opts.case)
        .ok_or("unknown case name")?;
    let (output, report) = if matches!(opts.backend.as_str(), "cuda" | "cudnn" | "nvidia") {
        #[cfg(feature = "cuda")]
        {
            run_cuda(case, &opts)?
        }
        #[cfg(not(feature = "cuda"))]
        {
            return Err("compile with --features cuda".into());
        }
    } else {
        run_cpu(case, &opts)?
    };
    if let Some(path) = &opts.output {
        let mut bytes = Vec::with_capacity(output.data().len() * 4);
        for value in output.data() {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        fs::write(path, bytes)?;
    }
    println!("{}", serde_json::to_string(&report)?);
    Ok(())
}

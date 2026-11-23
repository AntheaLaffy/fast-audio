//! Persistent pointwise measurement worker. See docs/pointwise-characterization-v2.md.
use fast_audio::{BinaryOp, Layout, Tensor, UnaryOp, cpu};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    fs,
    hint::black_box,
    io::{self, BufRead, Write},
    path::{Path, PathBuf},
    time::Instant,
};
type AnyResult<T> = Result<T, Box<dyn std::error::Error>>;

#[derive(Clone, Deserialize, Serialize)]
struct Geometry {
    dims: Vec<usize>,
    strides: Vec<isize>,
    offset: usize,
}
impl Geometry {
    fn layout(&self) -> fast_audio::Result<Layout> {
        Layout::from_strides(&self.dims, &self.strides, self.offset)
    }
}
#[derive(Clone, Deserialize, Serialize)]
struct Case {
    name: String,
    operation: String,
    regime: String,
    inputs: Vec<Geometry>,
    output: Geometry,
}
#[derive(Clone, Copy, Deserialize, Serialize)]
struct Window {
    warmup: usize,
    iterations: usize,
}
#[derive(Deserialize)]
struct Stratum {
    blocks: usize,
    windows: BTreeMap<String, Window>,
}
#[derive(Deserialize)]
struct Protocol {
    schema_version: u32,
    name: String,
    strata: BTreeMap<String, Stratum>,
    profiles: BTreeMap<String, BTreeMap<String, usize>>,
    cases: Vec<Case>,
}
struct Options {
    device: String,
    protocol: PathBuf,
    case: String,
    output: PathBuf,
    profile: bool,
}
fn options() -> AnyResult<Options> {
    let mut out = Options {
        device: "cpu".into(),
        protocol: "benchmarks/protocols/pointwise-characterization-v2.json".into(),
        case: String::new(),
        output: PathBuf::new(),
        profile: false,
    };
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--profile" {
            out.profile = true;
            continue;
        }
        if arg == "--help" {
            println!(
                "fa-pointwise --device cpu|cuda --case NAME --output-dir NEW_DIR [--protocol JSON] [--profile]"
            );
            std::process::exit(0);
        }
        let value = args.next().ok_or("option requires a value")?;
        match arg.as_str() {
            "--device" => out.device = value,
            "--case" => out.case = value,
            "--protocol" => out.protocol = value.into(),
            "--output-dir" => out.output = value.into(),
            _ => return Err(format!("unknown option {arg}").into()),
        }
    }
    if !matches!(out.device.as_str(), "cpu" | "cuda")
        || out.case.is_empty()
        || out.output.as_os_str().is_empty()
        || out.output.exists()
    {
        return Err("use cpu|cuda, an explicit case and a new output directory".into());
    }
    Ok(out)
}
fn unary(name: &str) -> Option<UnaryOp> {
    Some(match name {
        "neg" => UnaryOp::Neg,
        "abs" => UnaryOp::Abs,
        "square" => UnaryOp::Square,
        "reciprocal" => UnaryOp::Reciprocal,
        "sqrt" => UnaryOp::Sqrt,
        "exp" => UnaryOp::Exp,
        "log" => UnaryOp::Log,
        "relu" => UnaryOp::Relu,
        "sigmoid" => UnaryOp::Sigmoid,
        "tanh" => UnaryOp::Tanh,
        "silu" => UnaryOp::Silu,
        "softplus" => UnaryOp::Softplus,
        _ => return None,
    })
}
fn binary(name: &str) -> Option<BinaryOp> {
    Some(match name {
        "add" => BinaryOp::Add,
        "sub" => BinaryOp::Sub,
        "mul" => BinaryOp::Mul,
        "div" => BinaryOp::Div,
        _ => return None,
    })
}
fn data(layout: &Layout, salt: u32, kind: usize) -> fast_audio::Result<Tensor> {
    let size = layout
        .span()
        .map_or(layout.offset(), |s| s.end)
        .checked_add(2)
        .ok_or_else(|| fast_audio::Error::Invalid("fixture length overflows".into()))?;
    let mut tensor = Tensor::zeros([size])?;
    for (i, value) in tensor.data_mut().iter_mut().enumerate() {
        let state = (i as u32).wrapping_mul(747796405).wrapping_add(salt);
        let word = ((state >> ((state >> 28) + 4)) ^ state).wrapping_mul(277803737);
        let bits = ((word >> 22) ^ word) & 65535;
        let x = (bits as i32 - 32768) as f32 / 32768.;
        *value = match kind {
            1 => 0.125 + x.abs(),
            2 => 0.5 + x.abs(),
            _ => 4. * x,
        };
    }
    Ok(tensor)
}
struct Inputs {
    a: Tensor,
    b: Option<Tensor>,
    out: Tensor,
}
fn inputs(case: &Case, layouts: &[Layout], out: &Layout) -> fast_audio::Result<Inputs> {
    let a = data(
        &layouts[0],
        11,
        usize::from(matches!(
            case.operation.as_str(),
            "reciprocal" | "sqrt" | "log"
        )),
    )?;
    let b = if layouts.len() == 2 {
        Some(data(&layouts[1], 29, 2)?)
    } else {
        None
    };
    let n = out.span().map_or(out.offset(), |s| s.end) + 2;
    Ok(Inputs {
        a,
        b,
        out: Tensor::from_vec([n], vec![f32::from_bits(0x7fc12345); n])?,
    })
}
fn store(path: &Path, tensor: &Tensor) -> io::Result<()> {
    let mut bytes = Vec::with_capacity(tensor.data().len() * 4);
    for &value in tensor.data() {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    fs::write(path, bytes)
}
fn dump_host(directory: &Path, inputs: &Inputs) -> AnyResult<()> {
    store(&directory.join("input-0.f32"), &inputs.a)?;
    if let Some(b) = &inputs.b {
        store(&directory.join("input-1.f32"), b)?;
    }
    store(&directory.join("native-out.f32"), &inputs.out)?;
    Ok(())
}
enum HostOp {
    Unary(cpu::UnaryPlan),
    Binary(cpu::BinaryPlan),
}
impl HostOp {
    fn new(case: &Case, input: &[Layout], output: &Layout) -> fast_audio::Result<Self> {
        if let Some(op) = unary(&case.operation) {
            Ok(Self::Unary(cpu::UnaryPlan::new(op, &input[0], output)?))
        } else {
            Ok(Self::Binary(cpu::BinaryPlan::new(
                binary(&case.operation).expect("validated operation"),
                &input[0],
                &input[1],
                output,
            )?))
        }
    }
    fn run(&self, inputs: &mut Inputs) -> fast_audio::Result<()> {
        let Inputs { a, b, out } = inputs;
        // Black-box payload borrows per call so LTO cannot eliminate repeated pure writes.
        let a = black_box(&*a);
        let out = black_box(out);
        match self {
            Self::Unary(p) => p.run(
                &a.view(p.source_layout())?,
                &mut out.view_mut(p.destination_layout())?,
            ),
            Self::Binary(p) => p.run(
                &a.view(p.left_layout())?,
                &black_box(b.as_ref().unwrap()).view(p.right_layout())?,
                &mut out.view_mut(p.destination_layout())?,
            ),
        }
    }
    fn metadata(&self) -> Value {
        let (strategy, rank, inner, host, workspace) = match self {
            Self::Unary(p) => (
                p.strategy(),
                p.coalesced_rank(),
                p.inner_len(),
                p.host_metadata_bytes(),
                p.workspace_bytes(),
            ),
            Self::Binary(p) => (
                p.strategy(),
                p.coalesced_rank(),
                p.inner_len(),
                p.host_metadata_bytes(),
                p.workspace_bytes(),
            ),
        };
        json!({"strategy":format!("{strategy:?}"),"coalesced_rank":rank,"inner_len":inner,"host_metadata_bytes":host,"workspace_bytes":workspace})
    }
}
#[cfg(feature = "cuda")]
mod device {
    use super::*;
    use fast_audio::cuda::{self, Cuda, DeviceTensor};
    unsafe extern "C" {
        fn cudaProfilerStart() -> i32;
        fn cudaProfilerStop() -> i32;
    }
    enum Op {
        Unary(cuda::UnaryPlan),
        Binary(cuda::BinaryPlan),
    }
    pub(super) struct Gpu {
        pub cuda: Cuda,
        op: Op,
        a: DeviceTensor,
        b: Option<DeviceTensor>,
        out: DeviceTensor,
    }
    impl Gpu {
        pub fn new(
            case: &Case,
            layouts: &[Layout],
            output: &Layout,
            host: Inputs,
        ) -> AnyResult<(Self, f64, f64)> {
            let start = Instant::now();
            let cuda = Cuda::new(0)?;
            let a = cuda.upload(&host.a)?;
            let b = host.b.as_ref().map(|t| cuda.upload(t)).transpose()?;
            let out = cuda.upload(&host.out)?;
            cuda.synchronize()?;
            let payload = start.elapsed().as_secs_f64() * 1e6;
            let start = Instant::now();
            let op = if let Some(op) = unary(&case.operation) {
                Op::Unary(cuda::UnaryPlan::new(&cuda, op, &layouts[0], output)?)
            } else {
                Op::Binary(cuda::BinaryPlan::new(
                    &cuda,
                    binary(&case.operation).unwrap(),
                    &layouts[0],
                    &layouts[1],
                    output,
                )?)
            };
            cuda.synchronize()?;
            let plan = start.elapsed().as_secs_f64() * 1e6;
            Ok((
                Self {
                    cuda,
                    op,
                    a,
                    b,
                    out,
                },
                payload,
                plan,
            ))
        }
        pub fn run(&mut self) -> fast_audio::Result<()> {
            match &self.op {
                Op::Unary(p) => p.run(
                    &black_box(&self.a).view(p.source_layout())?,
                    &mut black_box(&mut self.out).view_mut(p.destination_layout())?,
                ),
                Op::Binary(p) => p.run(
                    &black_box(&self.a).view(p.left_layout())?,
                    &black_box(self.b.as_ref().unwrap()).view(p.right_layout())?,
                    &mut black_box(&mut self.out).view_mut(p.destination_layout())?,
                ),
            }
        }
        pub fn measure(&mut self, count: usize) -> fast_audio::Result<cuda::Timing> {
            let cuda = self.cuda.clone();
            cuda.measure(count, || self.run())
        }
        pub fn dump(&self, directory: &Path) -> AnyResult<()> {
            store(
                &directory.join("input-0.f32"),
                &self.cuda.download(&self.a)?,
            )?;
            if let Some(b) = &self.b {
                store(&directory.join("input-1.f32"), &self.cuda.download(b)?)?;
            }
            store(
                &directory.join("native-out.f32"),
                &self.cuda.download(&self.out)?,
            )?;
            Ok(())
        }
        pub fn metadata(&self) -> Value {
            let (strategy, rank, inner, host, workspace) = match &self.op {
                Op::Unary(p) => (
                    p.strategy(),
                    p.coalesced_rank(),
                    p.inner_len(),
                    p.host_metadata_bytes(),
                    p.workspace_bytes(),
                ),
                Op::Binary(p) => (
                    p.strategy(),
                    p.coalesced_rank(),
                    p.inner_len(),
                    p.host_metadata_bytes(),
                    p.workspace_bytes(),
                ),
            };
            json!({"strategy":format!("{strategy:?}"),"coalesced_rank":rank,"inner_len":inner,"host_metadata_bytes":host,"workspace_bytes":workspace})
        }
        pub fn profiler(start: bool) -> AnyResult<()> {
            // SAFETY: CUDA runtime profiler controls take no pointers and run only in
            // the explicit diagnostic path; the calling worker retains its context.
            let status = unsafe {
                if start {
                    cudaProfilerStart()
                } else {
                    cudaProfilerStop()
                }
            };
            if status != 0 {
                return Err(format!("CUDA profiler control status {status}").into());
            }
            Ok(())
        }
    }
}
enum Worker {
    Cpu {
        op: HostOp,
        inputs: Inputs,
    },
    #[cfg(feature = "cuda")]
    Gpu(device::Gpu),
}
impl Worker {
    fn run(&mut self) -> fast_audio::Result<()> {
        match self {
            Self::Cpu { op, inputs } => op.run(inputs),
            #[cfg(feature = "cuda")]
            Self::Gpu(g) => g.run(),
        }
    }
    fn sync(&self) -> fast_audio::Result<()> {
        match self {
            Self::Cpu { .. } => Ok(()),
            #[cfg(feature = "cuda")]
            Self::Gpu(g) => g.cuda.synchronize(),
        }
    }
    fn dump(&self, directory: &Path) -> AnyResult<()> {
        match self {
            Self::Cpu { inputs, .. } => dump_host(directory, inputs),
            #[cfg(feature = "cuda")]
            Self::Gpu(g) => g.dump(directory),
        }
    }
    fn metadata(&self) -> Value {
        match self {
            Self::Cpu { op, .. } => op.metadata(),
            #[cfg(feature = "cuda")]
            Self::Gpu(g) => g.metadata(),
        }
    }
    fn measure(&mut self, count: usize) -> fast_audio::Result<Value> {
        #[cfg(feature = "cuda")]
        if let Self::Gpu(g) = self {
            let t = g.measure(count)?;
            return Ok(json!({"wall_us":t.wall_us,"device_us":t.device_us}));
        }
        let start = Instant::now();
        for _ in 0..count {
            self.run()?;
        }
        Ok(json!({"wall_us":start.elapsed().as_secs_f64()*1e6/count as f64,"device_us":null}))
    }
    fn profiler(&self, _start: bool) -> AnyResult<()> {
        #[cfg(feature = "cuda")]
        if let Self::Gpu(_) = self {
            device::Gpu::profiler(_start)?;
        }
        Ok(())
    }
}
fn respond(writer: &mut impl Write, value: &Value) -> AnyResult<()> {
    serde_json::to_writer(&mut *writer, value)?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    Ok(())
}
fn main() -> AnyResult<()> {
    let opts = options()?;
    let protocol: Protocol = serde_json::from_slice(&fs::read(&opts.protocol)?)?;
    if protocol.schema_version != 1 || protocol.name != "pointwise-characterization-v2" {
        return Err("wrong protocol".into());
    }
    let case = protocol
        .cases
        .iter()
        .find(|c| c.name == opts.case)
        .ok_or("unknown case")?;
    let arity = if unary(&case.operation).is_some() {
        1
    } else if binary(&case.operation).is_some() {
        2
    } else {
        return Err("unknown operation".into());
    };
    if case.inputs.len() != arity {
        return Err("wrong input arity".into());
    }
    let stratum = if opts.device == "cpu" {
        "intel"
    } else {
        "nvidia"
    };
    let window = protocol.strata[stratum].windows[&case.regime];
    if window.iterations == 0 || protocol.strata[stratum].blocks == 0 {
        return Err("invalid timing counts".into());
    }
    let start = Instant::now();
    let layouts: Vec<_> = case
        .inputs
        .iter()
        .map(Geometry::layout)
        .collect::<fast_audio::Result<_>>()?;
    let output = case.output.layout()?;
    let host = inputs(case, &layouts, &output)?;
    let host_prep = start.elapsed().as_secs_f64() * 1e6;
    let (mut worker, payload_prep_us, plan_setup_us) = if opts.device == "cpu" {
        let start = Instant::now();
        let op = HostOp::new(case, &layouts, &output)?;
        (
            Worker::Cpu { op, inputs: host },
            host_prep,
            start.elapsed().as_secs_f64() * 1e6,
        )
    } else {
        #[cfg(feature = "cuda")]
        {
            let (g, payload, plan) = device::Gpu::new(case, &layouts, &output, host)?;
            (Worker::Gpu(g), host_prep + payload, plan)
        }
        #[cfg(not(feature = "cuda"))]
        {
            return Err("compile with --features cuda".into());
        }
    };
    let first = Instant::now();
    worker.run()?;
    worker.sync()?;
    let first_call_us = first.elapsed().as_secs_f64() * 1e6;
    fs::create_dir_all(&opts.output)?;
    worker.dump(&opts.output)?;
    let features = json!({"mkl":cfg!(feature="mkl"),"onednn":cfg!(feature="onednn"),"cuda":cfg!(feature="cuda"),"cudnn":cfg!(feature="cudnn")});
    let mut writer = io::BufWriter::new(io::stdout().lock());
    let process_status = fs::read_to_string("/proc/self/status")?;
    let affinity = process_status
        .lines()
        .find_map(|line| line.strip_prefix("Cpus_allowed_list:"))
        .ok_or("missing worker affinity")?
        .trim();
    respond(
        &mut writer,
        &json!({"status":"ready","engine":"native","case":case,"device":opts.device,"compiled_features":features,
        "debug_assertions":cfg!(debug_assertions),"cpu_allowed_list":affinity,"payload_prep_us":payload_prep_us,"plan_setup_us":plan_setup_us,"first_call_us":first_call_us,
        "window":window,"metadata":worker.metadata()}),
    )?;
    if opts.profile {
        let count = *protocol.profiles[stratum]
            .get(&case.name)
            .ok_or("case has no declared profile")?;
        for _ in 0..window.warmup {
            worker.run()?;
        }
        worker.sync()?;
        worker.profiler(true)?;
        let timing = worker.measure(count)?;
        worker.profiler(false)?;
        worker.dump(&opts.output)?;
        respond(
            &mut writer,
            &json!({"status":"profiled","case":case.name,"iterations":count,"warmup":window.warmup,"timing":timing}),
        )?;
        return Ok(());
    }
    let mut next = 0;
    for line in io::stdin().lock().lines() {
        let request: Value = serde_json::from_str(&line?)?;
        match request["action"].as_str() {
            Some("quit") => {
                respond(
                    &mut writer,
                    &json!({"status":"stopped","measured_blocks":next}),
                )?;
                return Ok(());
            }
            Some("measure")
                if request["mode"] == "native"
                    && request["block"].as_u64() == Some(next)
                    && next < protocol.strata[stratum].blocks as u64 =>
            {
                for _ in 0..window.warmup {
                    worker.run()?;
                }
                worker.sync()?;
                let timing = worker.measure(window.iterations)?;
                worker.dump(&opts.output)?;
                respond(
                    &mut writer,
                    &json!({"status":"measured","mode":"native","case":case.name,"block":next,"window":window,"timing":timing}),
                )?;
                next += 1;
            }
            _ => return Err("unexpected worker request or block sequence".into()),
        }
    }
    Ok(())
}

//! Resident complete-training worker; see docs/training-characterization-v1.md.
use fast_audio::{
    BinaryOp as B, Layout, ReduceOp, Shared, Tensor, UnaryOp,
    autograd::{Binding, Builder, PrepareOptions, Program, Resources, Seed, Value as Node},
    cpu::{self, Cpu},
};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    alloc::{GlobalAlloc, Layout as AllocLayout, System},
    cell::Cell,
    collections::BTreeMap,
    fs,
    hint::black_box,
    io::{self, BufRead, Write},
    path::{Path, PathBuf},
    time::Instant,
};
type AnyResult<T> = Result<T, Box<dyn std::error::Error>>;
const DECLARATION: &str =
    include_str!("../../benchmarks/protocols/training-characterization-v1.json");
thread_local! { static ALLOCATIONS: Cell<Option<usize>> = const { Cell::new(None) }; }
struct Allocator;
// SAFETY: allocation/deallocation arguments are passed unchanged to System.
unsafe impl GlobalAlloc for Allocator {
    unsafe fn alloc(&self, layout: AllocLayout) -> *mut u8 {
        ALLOCATIONS.with(|c| {
            if let Some(n) = c.get() {
                c.set(Some(n + 1));
            }
        });
        // SAFETY: forwards the caller's valid allocator layout.
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: AllocLayout) {
        // SAFETY: forwards the original pointer/layout pair to its allocator.
        unsafe { System.dealloc(pointer, layout) }
    }
}
#[global_allocator]
static ALLOCATOR: Allocator = Allocator;

#[derive(Clone, Deserialize)]
struct Case {
    name: String,
    family: String,
    dims: Vec<usize>,
    layout: String,
    regime: String,
}
#[derive(Clone, Copy, Deserialize)]
struct Window {
    warmup: usize,
    iterations: usize,
}
#[derive(Deserialize)]
struct Stratum {
    blocks: usize,
}
#[derive(Deserialize)]
struct Protocol {
    cases: Vec<Case>,
    windows: BTreeMap<String, Window>,
    strata: BTreeMap<String, Stratum>,
    profiles: BTreeMap<String, BTreeMap<String, usize>>,
    validation_steps: usize,
    rate: f32,
}
struct Options {
    platform: String,
    case: String,
    protocol: PathBuf,
    output: PathBuf,
    profile: bool,
    captured: bool,
}
fn options() -> AnyResult<Options> {
    let mut out = Options {
        platform: String::new(),
        case: String::new(),
        protocol: "benchmarks/protocols/training-characterization-v1.json".into(),
        output: PathBuf::new(),
        profile: false,
        captured: false,
    };
    let mut args = std::env::args().skip(1);
    while let Some(key) = args.next() {
        if key == "--profile" {
            out.profile = true;
            continue;
        }
        if key == "--captured" {
            out.captured = true;
            continue;
        }
        if key == "--help" {
            println!(
                "fa-training --platform intel|nvidia --case NAME --output-dir NEW_DIR [--protocol JSON] [--profile] [--captured]"
            );
            std::process::exit(0);
        }
        let value = args.next().ok_or("option requires a value")?;
        match key.as_str() {
            "--platform" => out.platform = value,
            "--case" => out.case = value,
            "--protocol" => out.protocol = value.into(),
            "--output-dir" => out.output = value.into(),
            _ => return Err(format!("unknown option {key}").into()),
        }
    }
    if !matches!(out.platform.as_str(), "intel" | "nvidia")
        || out.case.is_empty()
        || out.output.as_os_str().is_empty()
        || out.output.exists()
    {
        return Err("use an explicit platform/case and a new output directory".into());
    }
    Ok(out)
}

struct Fixture {
    program: Program,
    roots: [Node; 2],
    leaves: Vec<Node>,
    parameters: Vec<Node>,
    inputs: [Tensor; 2],
    initial: Vec<Tensor>,
    seed_layouts: [Layout; 2],
}
fn data(dims: &[usize], recipe: usize) -> fast_audio::Result<Tensor> {
    let mut out = Tensor::zeros(dims)?;
    for (i, v) in out.data_mut().iter_mut().enumerate() {
        *v = match recipe {
            0 => (((i * 7 + 3) % 29) as f32 - 14.) / 32.,
            1 => (((i * 5 + 1) % 23) as f32 - 11.) / 32.,
            2 => (((i * 3 + 2) % 17) as f32 + 4.) / 64.,
            3 => (((i * 13 + 5) % 31) as f32 - 15.) / 256.,
            4 => 0.5 + (i % 7) as f32 / 32.,
            5 => (((i * 3 + 2) % 7) as f32 - 3.) / 64.,
            _ => unreachable!(),
        };
    }
    Ok(out)
}
fn fixture(case: &Case) -> AnyResult<Fixture> {
    let (input_dims, target_dims, initial) = match (case.family.as_str(), case.dims.as_slice()) {
        ("dense", &[m, k, n]) => (vec![m, k], vec![m, n], vec![data(&[k, n], 3)?]),
        ("waveform", &[b, c, t]) => (
            vec![b, c, t],
            if case.layout == "contiguous" {
                vec![b, c, t]
            } else {
                vec![b, t, c]
            },
            vec![data(&[c], 4)?, data(&[c], 5)?],
        ),
        ("spectrum", &[b, c, f, t]) => (
            vec![b, c, f, t, 2],
            if case.layout == "contiguous" {
                vec![b, c, f, t]
            } else {
                vec![b, t, c, f]
            },
            vec![
                data(&[c, f], 4)?,
                Tensor::from_vec([2], vec![-0.0625, 0.125])?,
            ],
        ),
        _ => return Err("invalid declared training family/geometry".into()),
    };
    let inputs = [
        data(&input_dims, 0)?,
        data(&target_dims, if case.family == "spectrum" { 2 } else { 1 })?,
    ];
    let mut builder = Builder::new()?;
    let x = builder.leaf(&input_dims, false)?;
    let target = builder.leaf(&target_dims, false)?;
    let parameters = initial
        .iter()
        .map(|t| builder.leaf(t.shape().dims(), true))
        .collect::<fast_audio::Result<Vec<_>>>()?;
    let prediction = if case.family == "dense" {
        builder.matmul(x, parameters[0])?
    } else {
        let c = case.dims[1];
        let gain_dims = if case.family == "waveform" {
            vec![1, c, 1]
        } else {
            vec![1, c, case.dims[2], 1, 1]
        };
        let gain = builder.reshape(parameters[0], gain_dims)?;
        let bias = if case.family == "waveform" {
            builder.reshape(parameters[1], [1, c, 1])?
        } else {
            parameters[1]
        };
        let scaled = builder.binary(B::Mul, x, gain)?;
        let shifted = builder.binary(B::Add, scaled, bias)?;
        let mut prediction = if case.family == "waveform" {
            builder.unary(UnaryOp::Tanh, shifted)?
        } else {
            let square = builder.unary(UnaryOp::Square, shifted)?;
            builder.reduce(ReduceOp::Sum, square, &[4], false)?
        };
        if case.layout == "reversed-permuted" {
            let axis = if case.family == "waveform" { 2 } else { 3 };
            let t = case.dims[axis];
            prediction = builder.slice(prediction, axis, t - 1, t, -1)?;
            prediction = builder.permute(
                prediction,
                if axis == 2 { &[0, 2, 1] } else { &[0, 3, 1, 2] },
            )?;
        }
        prediction
    };
    let residual = builder.binary(B::Sub, prediction, target)?;
    let squared = builder.unary(UnaryOp::Square, residual)?;
    let axes = (0..target_dims.len()).collect::<Vec<_>>();
    let loss = builder.reduce(ReduceOp::Mean, squared, &axes, false)?;
    let detached = builder.detach(prediction)?;
    let roots = [loss, detached];
    let mut leaves = vec![x, target];
    leaves.extend_from_slice(&parameters);
    Ok(Fixture {
        program: builder.finish(&roots)?,
        roots,
        leaves,
        parameters,
        inputs,
        initial,
        seed_layouts: [
            Layout::contiguous([])?,
            Layout::from_strides(&target_dims, vec![0; target_dims.len()], 0)?,
        ],
    })
}
fn resources(r: Resources) -> Value {
    json!({"activation_bytes":r.activation_bytes,"gradient_bytes":r.gradient_bytes,
        "edge_scratch_bytes":r.edge_scratch_bytes,"scalar_seed_bytes":r.scalar_seed_bytes,
        "operator_workspace_bytes":r.operator_workspace_bytes,"view_index_bytes":r.view_index_bytes,
        "operator_host_metadata_bytes":r.operator_host_metadata_bytes,"max_static_shared_bytes_per_block":r.max_static_shared_bytes_per_block})
}
fn intel_cpu() -> fast_audio::Result<Cpu> {
    #[cfg(feature = "mkl")]
    {
        Cpu::mkl(1)
    }
    #[cfg(not(feature = "mkl"))]
    {
        Err(fast_audio::Error::Unsupported(
            "Intel training requires the mkl feature".into(),
        ))
    }
}
fn store(directory: &Path, name: &str, tensor: &Tensor) -> AnyResult<Value> {
    let path = directory.join(format!("{name}.f32"));
    let mut bytes = Vec::with_capacity(tensor.data().len() * 4);
    for value in tensor.data() {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    fs::write(&path, bytes)?;
    Ok(json!({"file":path,"shape":tensor.shape().dims()}))
}
struct Host {
    tape: cpu::Tape,
    owners: Vec<Shared<Tensor>>,
    fixture: Fixture,
    seeds: [Tensor; 2],
    rate: f32,
}
impl Host {
    fn new(fixture: Fixture, cpu: Cpu, rate: f32) -> AnyResult<Self> {
        let owners = fixture
            .inputs
            .iter()
            .chain(&fixture.initial)
            .map(|t| Shared::new(t.clone()))
            .collect::<fast_audio::Result<Vec<_>>>()?;
        let bindings = fixture
            .leaves
            .iter()
            .zip(&owners)
            .map(|(&v, t)| Binding::new(v, t.clone()))
            .collect::<Vec<_>>();
        let tape = cpu::Tape::with_options(
            cpu,
            &fixture.program,
            &bindings,
            PrepareOptions {
                seed_layouts: Some(&fixture.seed_layouts),
                ..PrepareOptions::default()
            },
        )?;
        Ok(Self {
            tape,
            owners,
            fixture,
            seeds: [Tensor::from_vec([], vec![1.])?, Tensor::zeros([])?],
            rate,
        })
    }
    fn reset(&mut self) -> fast_audio::Result<()> {
        for (owner, initial) in self.owners[2..].iter().zip(&self.fixture.initial) {
            owner.write()?.data_mut().copy_from_slice(initial.data());
        }
        Ok(())
    }
    fn step(&mut self) -> fast_audio::Result<()> {
        let seeds = [
            Seed::new(
                self.fixture.roots[0],
                &self.seeds[0],
                &self.fixture.seed_layouts[0],
            ),
            Seed::new(
                self.fixture.roots[1],
                &self.seeds[1],
                &self.fixture.seed_layouts[1],
            ),
        ];
        self.tape.step(&seeds, &self.fixture.parameters, self.rate)
    }
    fn dump(&mut self, directory: &Path) -> AnyResult<Value> {
        self.tape.forward()?;
        let prediction = self
            .tape
            .output(self.fixture.roots[1])?
            .view()?
            .to_tensor()?;
        let loss = self.tape.output(self.fixture.roots[0])?.storage().clone();
        let mut out = serde_json::Map::new();
        out.insert(
            "prediction".into(),
            store(directory, "prediction", &prediction)?,
        );
        out.insert("loss".into(), store(directory, "loss", &loss)?);
        for (i, owner) in self.owners.iter().enumerate() {
            out.insert(
                format!("owner-{i}"),
                store(directory, &format!("owner-{i}"), &*owner.read()?)?,
            );
        }
        for (i, &v) in self.fixture.parameters.iter().enumerate() {
            out.insert(
                format!("gradient-{i}"),
                store(
                    directory,
                    &format!("gradient-{i}"),
                    self.tape.gradient(v)?.ok_or("missing native gradient")?,
                )?,
            );
        }
        Ok(Value::Object(out))
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
    pub(super) struct Gpu {
        pub cuda: Cuda,
        pub tape: cuda::Tape,
        owners: Vec<Shared<DeviceTensor>>,
        fixture: Fixture,
        seeds: [DeviceTensor; 2],
        rate: f32,
        captured: bool,
    }
    impl Gpu {
        pub fn new(fixture: Fixture, rate: f32, captured: bool) -> AnyResult<Self> {
            let cuda = Cuda::new(0)?;
            let owners = fixture
                .inputs
                .iter()
                .chain(&fixture.initial)
                .map(|t| Shared::new(cuda.upload(t)?))
                .collect::<fast_audio::Result<Vec<_>>>()?;
            let bindings = fixture
                .leaves
                .iter()
                .zip(&owners)
                .map(|(&v, t)| Binding::new(v, t.clone()))
                .collect::<Vec<_>>();
            let tape = cuda::Tape::with_options(
                &cuda,
                &fixture.program,
                &bindings,
                PrepareOptions {
                    seed_layouts: Some(&fixture.seed_layouts),
                    ..PrepareOptions::default()
                },
            )?;
            let seeds = [
                cuda.upload(&Tensor::from_vec([], vec![1.])?)?,
                cuda.zeros([])?,
            ];
            cuda.synchronize()?;
            Ok(Self {
                cuda,
                tape,
                owners,
                fixture,
                seeds,
                rate,
                captured,
            })
        }
        pub fn reset(&mut self) -> fast_audio::Result<()> {
            for (owner, initial) in self.owners[2..].iter().zip(&self.fixture.initial) {
                self.cuda.copy_from_host(initial, &mut *owner.write()?)?;
            }
            self.cuda.synchronize()
        }
        pub fn step(&mut self) -> fast_audio::Result<()> {
            let seeds = [
                Seed::new(
                    self.fixture.roots[0],
                    &self.seeds[0],
                    &self.fixture.seed_layouts[0],
                ),
                Seed::new(
                    self.fixture.roots[1],
                    &self.seeds[1],
                    &self.fixture.seed_layouts[1],
                ),
            ];
            if self.captured {
                self.tape
                    .step_captured(&seeds, &self.fixture.parameters, self.rate)
            } else {
                self.tape.step(&seeds, &self.fixture.parameters, self.rate)
            }
        }
        pub fn dump(&mut self, directory: &Path) -> AnyResult<Value> {
            self.tape.forward()?;
            self.tape.synchronize()?;
            let prediction = {
                let out = self.tape.output(self.fixture.roots[1])?;
                self.cuda
                    .download(out.storage())?
                    .view(out.layout())?
                    .to_tensor()?
            };
            let loss = self
                .cuda
                .download(self.tape.output(self.fixture.roots[0])?.storage())?;
            let mut out = serde_json::Map::new();
            out.insert(
                "prediction".into(),
                store(directory, "prediction", &prediction)?,
            );
            out.insert("loss".into(), store(directory, "loss", &loss)?);
            for (i, owner) in self.owners.iter().enumerate() {
                out.insert(
                    format!("owner-{i}"),
                    store(
                        directory,
                        &format!("owner-{i}"),
                        &self.cuda.download(&*owner.read()?)?,
                    )?,
                );
            }
            for (i, &v) in self.fixture.parameters.iter().enumerate() {
                out.insert(
                    format!("gradient-{i}"),
                    store(
                        directory,
                        &format!("gradient-{i}"),
                        &self
                            .cuda
                            .download(self.tape.gradient(v)?.ok_or("missing native gradient")?)?,
                    )?,
                );
            }
            Ok(Value::Object(out))
        }
        pub fn profiler(start: bool) -> AnyResult<()> {
            // SAFETY: pointer-free CUDA runtime profiler controls are used only in the
            // explicitly requested diagnostic path while the worker retains its session.
            let code = unsafe {
                if start {
                    cudaProfilerStart()
                } else {
                    cudaProfilerStop()
                }
            };
            if code != 0 {
                return Err(format!("CUDA profiler control failed: {code}").into());
            }
            Ok(())
        }
    }
}
enum Worker {
    Cpu(Host),
    #[cfg(feature = "cuda")]
    Gpu(device::Gpu),
}
impl Worker {
    fn step(&mut self) -> fast_audio::Result<()> {
        match self {
            Self::Cpu(h) => h.step(),
            #[cfg(feature = "cuda")]
            Self::Gpu(g) => g.step(),
        }
    }
    fn reset(&mut self) -> fast_audio::Result<()> {
        match self {
            Self::Cpu(h) => h.reset(),
            #[cfg(feature = "cuda")]
            Self::Gpu(g) => g.reset(),
        }
    }
    fn sync(&mut self) -> fast_audio::Result<()> {
        match self {
            Self::Cpu(h) => h.tape.synchronize(),
            #[cfg(feature = "cuda")]
            Self::Gpu(g) => g.tape.synchronize(),
        }
    }
    fn dump(&mut self, p: &Path) -> AnyResult<Value> {
        match self {
            Self::Cpu(h) => h.dump(p),
            #[cfg(feature = "cuda")]
            Self::Gpu(g) => g.dump(p),
        }
    }
    fn resources(&self) -> Resources {
        match self {
            Self::Cpu(h) => h.tape.resources(),
            #[cfg(feature = "cuda")]
            Self::Gpu(g) => g.tape.resources(),
        }
    }
    fn run(&mut self, count: usize) -> fast_audio::Result<()> {
        for _ in 0..count {
            black_box(&mut *self).step()?;
        }
        Ok(())
    }
    fn measure(&mut self, count: usize) -> fast_audio::Result<Value> {
        self.sync()?;
        ALLOCATIONS.with(|c| c.set(Some(0)));
        let measured = self.time(count);
        let allocations = ALLOCATIONS.with(|c| c.replace(None)).unwrap();
        let (wall, device) = measured?;
        if allocations != 0 {
            return Err(fast_audio::Error::Invalid(
                "warmed native training allocated Rust heap storage".into(),
            ));
        }
        Ok(json!({"wall_us":wall,"device_us":device,"hot_rust_allocations":allocations}))
    }
    fn time(&mut self, count: usize) -> fast_audio::Result<(f64, Option<f64>)> {
        match self {
            Self::Cpu(h) => {
                let start = Instant::now();
                for _ in 0..count {
                    black_box(&mut *h).step()?;
                }
                Ok((start.elapsed().as_secs_f64() * 1e6 / count as f64, None))
            }
            #[cfg(feature = "cuda")]
            Self::Gpu(g) => {
                let cuda = g.cuda.clone();
                let t = cuda.measure(count, || black_box(&mut *g).step())?;
                Ok((t.wall_us, Some(t.device_us)))
            }
        }
    }
}
fn respond(value: Value) -> AnyResult<()> {
    println!("{}", serde_json::to_string(&value)?);
    io::stdout().flush()?;
    Ok(())
}
fn main() -> AnyResult<()> {
    let args = options()?;
    if fs::read(&args.protocol)? != DECLARATION.as_bytes() {
        return Err("training v1 declaration differs from compiled bytes".into());
    }
    let protocol: Protocol = serde_json::from_str(DECLARATION)?;
    let case = protocol
        .cases
        .iter()
        .find(|c| c.name == args.case)
        .ok_or("unknown training case")?;
    let window = *protocol
        .windows
        .get(&case.regime)
        .ok_or("missing training window")?;
    let begin = Instant::now();
    let fixture = fixture(case)?;
    let bound_payload_bytes = fixture
        .inputs
        .iter()
        .chain(&fixture.initial)
        .map(|t| t.shape().len() * 4)
        .sum::<usize>();
    if args.platform == "nvidia"
        && std::env::var("CUBLAS_WORKSPACE_CONFIG").ok().as_deref() != Some(":4096:8")
    {
        return Err("NVIDIA training requires the declared CUBLAS_WORKSPACE_CONFIG".into());
    }
    let mut worker = match args.platform.as_str() {
        "intel" => Worker::Cpu(Host::new(fixture, intel_cpu()?, protocol.rate)?),
        "nvidia" => {
            #[cfg(feature = "cuda")]
            {
                Worker::Gpu(device::Gpu::new(fixture, protocol.rate, args.captured)?)
            }
            #[cfg(not(feature = "cuda"))]
            {
                let _ = fixture;
                return Err("NVIDIA training requires the cuda feature".into());
            }
        }
        _ => unreachable!(),
    };
    worker.sync()?;
    let preparation = begin.elapsed().as_secs_f64() * 1e6;
    let begin = Instant::now();
    worker.step()?;
    worker.sync()?;
    let first = begin.elapsed().as_secs_f64() * 1e6;
    fs::create_dir_all(&args.output)?;
    let snapshot = args.output.join("snapshot");
    fs::create_dir(&snapshot)?;
    respond(
        json!({"status":"ready","engine":"native","case":case.name,"platform":args.platform,
        "preparation_us":preparation,"first_step_us":first,"prepare_includes_execution":false,
        "resources":resources(worker.resources()),"seed_payload_bytes":8,"reset_source":"host_initial_fp32",
        "bound_payload_bytes":bound_payload_bytes,"host_fixture_bytes":bound_payload_bytes,
        "resource_scope":"Explicit bound/fixture/seed and tape payload/plan bytes; topology, control headers, allocator/vendor/process peaks excluded",
        "features":{"mkl":cfg!(feature="mkl"),"cuda":cfg!(feature="cuda")},"snapshot_semantics":"post-update forward; last pre-update gradient"}),
    )?;
    if args.profile {
        let count = *protocol
            .profiles
            .get(&args.platform)
            .and_then(|p| p.get(&case.name))
            .ok_or("case not declared for profiling")?;
        worker.reset()?;
        worker.run(window.warmup)?;
        worker.sync()?;
        #[cfg(feature = "cuda")]
        if args.platform == "nvidia" {
            device::Gpu::profiler(true)?;
        }
        worker.run(count)?;
        worker.sync()?;
        #[cfg(feature = "cuda")]
        if args.platform == "nvidia" {
            device::Gpu::profiler(false)?;
        }
        let dump = worker.dump(&snapshot)?;
        return respond(json!({"status":"profiled","iterations":count,"snapshot":dump}));
    }
    let mut validation = 0;
    let mut blocks = 0;
    let mut repeat = None;
    for line in io::stdin().lock().lines() {
        let request: Value = serde_json::from_str(&line?)?;
        let expected: &[&str] = match request["action"].as_str() {
            Some("quit") => &["action"],
            Some("validate") => &["action", "step"],
            Some("measure" | "repeat") => &["action", "block"],
            _ => return Err("unknown training request".into()),
        };
        if !request.as_object().is_some_and(|r| {
            r.len() == expected.len() && expected.iter().all(|k| r.contains_key(*k))
        }) {
            return Err("training request fields differ".into());
        }
        match request["action"].as_str() {
            Some("quit") => {
                respond(
                    json!({"status":"stopped","validation_steps":validation,"measured_blocks":blocks,"pending_repeat":repeat}),
                )?;
                return Ok(());
            }
            Some("validate")
                if request["step"].as_u64() == Some(validation)
                    && validation < protocol.validation_steps as u64
                    && blocks == 0 =>
            {
                if validation == 0 {
                    worker.reset()?;
                }
                worker.step()?;
                worker.sync()?;
                let dump = worker.dump(&snapshot)?;
                respond(json!({"status":"validated","step":validation,"snapshot":dump}))?;
                validation += 1;
            }
            Some("measure")
                if validation == protocol.validation_steps as u64
                    && repeat.is_none()
                    && request["block"].as_u64() == Some(blocks)
                    && blocks < protocol.strata[&args.platform].blocks as u64 =>
            {
                worker.reset()?;
                worker.run(window.warmup)?;
                worker.sync()?;
                let timing = worker.measure(window.iterations)?;
                let dump = worker.dump(&snapshot)?;
                respond(
                    json!({"status":"measured","block":blocks,"warmup":window.warmup,"iterations":window.iterations,"timing":timing,"snapshot":dump}),
                )?;
                repeat = Some(blocks);
                blocks += 1;
            }
            Some("repeat") if repeat.is_some() && request["block"].as_u64() == repeat => {
                worker.reset()?;
                worker.run(window.warmup + window.iterations)?;
                worker.sync()?;
                let dump = worker.dump(&snapshot)?;
                respond(json!({"status":"repeated","block":repeat,"snapshot":dump}))?;
                repeat = None;
            }
            _ => return Err("unexpected training action/order or declared budget exceeded".into()),
        }
    }
    Err("training worker input closed without quit".into())
}

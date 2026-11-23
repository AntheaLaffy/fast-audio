//! Replay actual GRU inputs and parameters captured by tools/gtcrn_reference.py.
use fast_audio::{
    GruSpec, GruWeights, Tensor,
    cpu::{self, Cpu},
};
use serde::Deserialize;
use serde_json::json;
use std::{collections::HashMap, fs, path::Path, time::Instant};

mod support;
use support::{AnyResult, Blob, load, save};
#[derive(Deserialize)]
struct Layer {
    name: String,
    #[serde(rename = "type")]
    kind: String,
    hidden_size: Option<usize>,
    bidirectional: Option<bool>,
    input: Option<Blob>,
    initial: Option<Blob>,
    parameters: Option<HashMap<String, Blob>>,
}
#[derive(Deserialize)]
struct Case {
    name: String,
    leaf_operators: Vec<Layer>,
}
#[derive(Deserialize)]
struct Reference {
    cases: Vec<Case>,
}
fn cpu_run(
    cpu: Cpu,
    s: GruSpec,
    p: &GruWeights,
    x: &Tensor,
    initial: &Tensor,
) -> AnyResult<(Tensor, Tensor, serde_json::Value)> {
    let begin = Instant::now();
    let mut plan = cpu::Gru::new(cpu, s, p)?;
    let setup = begin.elapsed().as_secs_f64() * 1e6;
    let mut y = Tensor::zeros(s.output_shape()?)?;
    let mut final_state = Tensor::zeros(s.state_shape())?;
    for _ in 0..20 {
        plan.run(x, initial, &mut y, &mut final_state)?;
    }
    let mut samples = Vec::new();
    for _ in 0..7 {
        let begin = Instant::now();
        for _ in 0..50 {
            plan.run(x, initial, &mut y, &mut final_state)?;
        }
        samples.push(begin.elapsed().as_secs_f64() * 1e6 / 50.0);
    }
    Ok((
        y,
        final_state,
        json!({"setup_us":setup,"wall_us":samples,"workspace_bytes":plan.workspace_bytes(),"parameter_bytes":plan.parameter_bytes()}),
    ))
}
#[cfg(feature = "cudnn")]
fn cuda_run(
    s: GruSpec,
    p: &GruWeights,
    x: &Tensor,
    initial: &Tensor,
) -> AnyResult<(Tensor, Tensor, serde_json::Value)> {
    use fast_audio::cuda;
    let gpu = cuda::Cuda::new(0)?;
    let x = gpu.upload(x)?;
    let initial = gpu.upload(initial)?;
    let mut y = gpu.zeros(s.output_shape()?)?;
    let mut final_state = gpu.zeros(s.state_shape())?;
    let begin = Instant::now();
    let mut plan = cuda::Gru::new(&gpu, s, p)?;
    let setup = begin.elapsed().as_secs_f64() * 1e6;
    for _ in 0..20 {
        plan.run(&x, &initial, &mut y, &mut final_state)?;
    }
    gpu.synchronize()?;
    let mut wall = Vec::new();
    let mut device = Vec::new();
    for _ in 0..7 {
        let sample = gpu.measure(50, || plan.run(&x, &initial, &mut y, &mut final_state))?;
        wall.push(sample.wall_us);
        device.push(sample.device_us);
    }
    Ok((
        gpu.download(&y)?,
        gpu.download(&final_state)?,
        json!({"implementation":plan.implementation(),"setup_us":setup,"wall_us":wall,"device_us":device,
        "workspace_bytes":plan.workspace_bytes(),"parameter_bytes":plan.parameter_bytes()}),
    ))
}
fn main() -> AnyResult<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 3 {
        return Err(
            "usage: gru-reference reference|mkl|cudnn REFERENCE_DIRECTORY OUTPUT_DIRECTORY".into(),
        );
    }
    let reference = Path::new(&args[1]);
    let out = Path::new(&args[2]);
    if out.exists() {
        return Err("use a new output directory".into());
    }
    fs::create_dir_all(out)?;
    let data: Reference = serde_json::from_slice(&fs::read(reference.join("reference.json"))?)?;
    let mut records = Vec::new();
    for case in data.cases {
        for layer in case.leaf_operators.into_iter().filter(|l| l.kind == "GRU") {
            let folder = reference.join(&case.name);
            let x = load(&folder, layer.input.as_ref().ok_or("missing GRU input")?)?;
            let initial = load(&folder, layer.initial.as_ref().ok_or("missing GRU state")?)?;
            let &[batch, steps, input_size] = x.shape().dims() else {
                return Err("GRU input must be batch-first rank 3".into());
            };
            let s = GruSpec {
                batch,
                steps,
                input_size,
                hidden_size: layer.hidden_size.ok_or("missing hidden size")?,
                bidirectional: layer.bidirectional.ok_or("missing direction")?,
            };
            let parameters = layer.parameters.as_ref().ok_or("missing GRU parameters")?;
            let stacked = |name: &str, shape: Vec<usize>| -> AnyResult<Tensor> {
                let mut values = Vec::new();
                for direction in 0..s.directions() {
                    let key = format!("{name}_l0{}", if direction == 0 { "" } else { "_reverse" });
                    values.extend_from_slice(
                        load(
                            &folder,
                            parameters.get(&key).ok_or("missing GRU parameter")?,
                        )?
                        .data(),
                    );
                }
                Ok(Tensor::from_vec(shape, values)?)
            };
            let p = GruWeights {
                input: stacked(
                    "weight_ih",
                    vec![s.directions(), 3 * s.hidden_size, s.input_size],
                )?,
                hidden: stacked(
                    "weight_hh",
                    vec![s.directions(), 3 * s.hidden_size, s.hidden_size],
                )?,
                input_bias: stacked("bias_ih", vec![s.directions(), 3 * s.hidden_size])?,
                hidden_bias: stacked("bias_hh", vec![s.directions(), 3 * s.hidden_size])?,
            };
            let (y, final_state, timing) = match args[0].as_str() {
                "reference" => cpu_run(Cpu::Reference, s, &p, &x, &initial)?,
                #[cfg(feature = "mkl")]
                "mkl" => cpu_run(Cpu::mkl(1)?, s, &p, &x, &initial)?,
                #[cfg(feature = "cudnn")]
                "cudnn" => cuda_run(s, &p, &x, &initial)?,
                _ => return Err("backend is unknown or was not enabled".into()),
            };
            let key = format!("{}-{}", case.name, layer.name.replace('.', "-"));
            let y_file = format!("{key}-output.f32");
            let state_file = format!("{key}-final.f32");
            save(&out.join(&y_file), &y)?;
            save(&out.join(&state_file), &final_state)?;
            records.push(json!({"case":case.name,"layer":layer.name,"output":{"file":y_file,"shape":y.shape().dims()},
            "final":{"file":state_file,"shape":final_state.shape().dims()},"timing":timing}));
        }
    }
    fs::write(
        out.join("gru.json"),
        serde_json::to_string_pretty(
            &json!({"schema_version":1,"backend":args[0],"cases":records}),
        )? + "\n",
    )?;
    Ok(())
}

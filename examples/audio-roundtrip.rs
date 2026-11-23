//! Emit complete spectra and reconstructed waveforms for tools/check_audio.py.
use fast_audio::{
    CenteredStftSpec, Error, Result, StftSpec, Tensor,
    cpu::{self, Cpu},
};
use serde_json::json;
use std::{fs, path::Path};

fn fixture(shape: impl AsRef<[usize]>, salt: usize) -> Result<Tensor> {
    let mut tensor = Tensor::zeros(shape)?;
    for (i, value) in tensor.data_mut().iter_mut().enumerate() {
        *value = ((i * 37 + salt) % 101) as f32 / 51.0 - 1.0;
    }
    Ok(tensor)
}
fn cpu_run(
    cpu: Cpu,
    s: CenteredStftSpec,
    x: &Tensor,
    modified: &Tensor,
) -> Result<(Tensor, Tensor, Tensor, usize)> {
    let mut spectrum = Tensor::zeros(s.output_shape())?;
    let mut restored = Tensor::zeros(s.input_shape())?;
    let mut processed = Tensor::zeros(s.input_shape())?;
    cpu::CenteredStft::new(cpu, s)?.run(x, &mut spectrum)?;
    let mut inverse = cpu::Istft::new(cpu, s)?;
    inverse.run(&spectrum, &mut restored)?;
    inverse.run(modified, &mut processed)?;
    Ok((spectrum, restored, processed, inverse.workspace_bytes()))
}
fn execute(
    backend: &str,
    s: CenteredStftSpec,
    x: &Tensor,
    modified: &Tensor,
) -> Result<(Tensor, Tensor, Tensor, usize)> {
    match backend {
        "reference" => cpu_run(Cpu::Reference, s, x, modified),
        #[cfg(feature = "mkl")]
        "mkl" => cpu_run(Cpu::mkl(1)?, s, x, modified),
        #[cfg(feature = "cuda")]
        "cuda" => {
            use fast_audio::cuda;
            let gpu = cuda::Cuda::new(0)?;
            let dx = gpu.upload(x)?;
            let modified = gpu.upload(modified)?;
            let mut spectrum = gpu.zeros(s.output_shape())?;
            let mut restored = gpu.zeros(s.input_shape())?;
            let mut processed = gpu.zeros(s.input_shape())?;
            cuda::CenteredStft::new(&gpu, s)?.run(&dx, &mut spectrum)?;
            let mut inverse = cuda::Istft::new(&gpu, s)?;
            inverse.run(&spectrum, &mut restored)?;
            inverse.run(&modified, &mut processed)?;
            Ok((
                gpu.download(&spectrum)?,
                gpu.download(&restored)?,
                gpu.download(&processed)?,
                inverse.workspace_bytes(),
            ))
        }
        _ => Err(Error::Invalid(format!(
            "backend {backend} is unknown or was not enabled at build time"
        ))),
    }
}
fn save(path: &Path, value: &Tensor) -> std::io::Result<()> {
    let bytes: Vec<u8> = value.data().iter().flat_map(|v| v.to_le_bytes()).collect();
    fs::write(path, bytes)
}
fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 2 {
        return Err("usage: audio-roundtrip reference|mkl|cuda OUTPUT_DIRECTORY".into());
    }
    let out = Path::new(&args[1]);
    if out.exists() {
        return Err("use a new output directory to preserve evidence".into());
    }
    fs::create_dir_all(out)?;
    let mut cases = Vec::new();
    for (batch, input_len, fft_size, hop) in [
        (2, 2, 2, 1),
        (2, 4, 7, 2),
        (2, 97, 15, 4),
        (2, 44101, 511, 128),
        (2, 44101, 512, 128),
        (2, 44101, 2048, 512),
    ] {
        let name = format!("b{batch}-l{input_len}-n{fft_size}-h{hop}");
        let s = StftSpec {
            batch,
            input_len,
            fft_size,
            hop,
        }
        .centered()?;
        let x = fixture(s.input_shape(), 11)?;
        let modified = fixture(s.output_shape(), 47)?;
        let (spectrum, restored, processed, workspace) = execute(&args[0], s, &x, &modified)?;
        let mut files = serde_json::Map::new();
        for (kind, tensor) in [
            ("input", x),
            ("spectrum", spectrum),
            ("restored", restored),
            ("modified", modified),
            ("processed", processed),
        ] {
            let file = format!("{name}-{kind}.f32");
            save(&out.join(&file), &tensor)?;
            files.insert(
                kind.into(),
                json!({"file":file, "shape":tensor.shape().dims()}),
            );
        }
        cases.push(
            json!({"name":name,"batch":batch,"input_len":input_len,"fft_size":fft_size,"hop":hop,
            "files":files,"inverse_workspace_bytes":workspace}),
        );
    }
    fs::write(
        out.join("audio.json"),
        serde_json::to_string_pretty(&json!({"schema_version":1,"backend":args[0],"cases":cases}))?
            + "\n",
    )?;
    Ok(())
}

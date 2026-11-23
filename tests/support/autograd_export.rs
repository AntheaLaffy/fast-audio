use fast_audio::{Tensor, cpu::Cpu};
use serde_json::json;
use std::io::Write;

pub fn backend(cpu: Cpu) -> &'static str {
    match cpu {
        Cpu::Reference => "reference",
        #[cfg(feature = "mkl")]
        Cpu::Mkl { .. } => "mkl",
    }
}
pub fn bits(t: &Tensor) -> Vec<u32> {
    t.data().iter().map(|v| v.to_bits()).collect()
}
pub fn tensor(t: &Tensor) -> serde_json::Value {
    json!({"shape":t.shape().dims(),"bits":bits(t)})
}
/// Optional, exclusive raw-bit records. Call only outside the allocation guards.
pub struct Exporter(Option<std::io::BufWriter<std::fs::File>>);
pub struct Composition<'a> {
    pub backend: &'a str,
    pub index: usize,
    pub binding: usize,
    pub inputs: &'a [Tensor; 2],
    pub seed: &'a Tensor,
    pub output: &'a Tensor,
    pub gradients: [&'a [Tensor; 2]; 2],
}
impl Exporter {
    pub fn new(role: &str) -> Self {
        Self(
            std::env::var_os("FAST_AUDIO_AUTOGRAD_EXPORT_DIR").map(|directory| {
                let directory = std::path::PathBuf::from(directory);
                std::fs::create_dir_all(&directory).unwrap();
                std::io::BufWriter::new(
                    std::fs::OpenOptions::new()
                        .write(true)
                        .create_new(true)
                        .open(directory.join(format!("{role}.jsonl")))
                        .unwrap(),
                )
            }),
        )
    }
    pub fn record(&mut self, value: serde_json::Value) {
        if let Some(writer) = &mut self.0 {
            serde_json::to_writer(&mut *writer, &value).unwrap();
            writer.write_all(b"\n").unwrap();
        }
    }
    pub fn finish(mut self) {
        if let Some(writer) = &mut self.0 {
            writer.flush().unwrap();
        }
    }
    pub fn composition(&mut self, sample: Composition<'_>) {
        if self.0.is_none() {
            return;
        }
        let Composition {
            backend,
            index,
            binding,
            inputs,
            seed,
            output,
            gradients,
        } = sample;
        self.record(
            json!({"schema_version":1,"kind":"composition","backend":backend,
            "case_index":index,"binding":binding,"inputs":inputs.each_ref().map(tensor),
            "seed":tensor(seed),"output":tensor(output),
            "gradients":gradients[0].each_ref().map(tensor),
            "accumulated":gradients[1].each_ref().map(tensor)}),
        );
    }
}

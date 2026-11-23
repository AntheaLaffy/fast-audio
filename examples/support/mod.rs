use fast_audio::Tensor;
use serde::Deserialize;
use std::{fs, path::Path};
pub type AnyResult<T> = std::result::Result<T, Box<dyn std::error::Error>>;
#[derive(Deserialize)]
pub struct Blob {
    pub file: String,
    pub shape: Vec<usize>,
}
pub fn load(folder: &Path, description: &Blob) -> AnyResult<Tensor> {
    let bytes = fs::read(folder.join(&description.file))?;
    if bytes.len() % 4 != 0 {
        return Err("invalid FP32 byte size".into());
    }
    Ok(Tensor::from_vec(
        &description.shape,
        bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect(),
    )?)
}
pub fn save(path: &Path, value: &Tensor) -> AnyResult<()> {
    fs::write(
        path,
        value
            .data()
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )?;
    Ok(())
}

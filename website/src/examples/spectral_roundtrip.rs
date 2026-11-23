use fast_audio::{Result, StftSpec, Tensor, cpu};

fn main() -> Result<()> {
    let spec = StftSpec {
        batch: 1,
        input_len: 64,
        fft_size: 16,
        hop: 4,
    }
    .centered()?;
    let samples = (0..64).map(|i| (i as f32 * 0.2).sin()).collect();
    let input = Tensor::from_vec(spec.input_shape(), samples)?;
    let mut spectrum = Tensor::zeros(spec.output_shape())?;
    let mut restored = Tensor::zeros(spec.input_shape())?;

    let backend = cpu::Cpu::Reference;
    cpu::CenteredStft::new(backend, spec)?.run(&input, &mut spectrum)?;
    cpu::Istft::new(backend, spec)?.run(&spectrum, &mut restored)?;

    let error = input
        .data()
        .iter()
        .zip(restored.data())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, f32::max);
    assert!(error < 1e-5);
    println!("Maximum reconstruction error: {error:.2e}");
    Ok(())
}

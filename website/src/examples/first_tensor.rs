use fast_audio::{Result, Tensor, cpu::Cpu};

fn main() -> Result<()> {
    let signal = Tensor::from_vec([2, 3], vec![1., 2., 3., 4., 5., 6.])?;
    let weights = Tensor::from_vec([3, 1], vec![1., 0., -1.])?;
    let mut output = Tensor::zeros([2, 1])?;

    Cpu::Reference.matmul(&signal, &weights, &mut output)?;
    assert_eq!(output.shape().dims(), &[2, 1]);
    assert_eq!(output.data(), &[-2., -2.]);
    println!("{:?}", output.data());
    Ok(())
}

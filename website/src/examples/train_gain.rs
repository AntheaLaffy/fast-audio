use fast_audio::{
    BinaryOp, ReduceOp, Result, Shared, Tensor, UnaryOp,
    autograd::{Binding, Builder},
    cpu::{Cpu, Tape},
};

fn main() -> Result<()> {
    let mut graph = Builder::new()?;
    let signal = graph.leaf([4], false)?;
    let target = graph.leaf([4], false)?;
    let gain = graph.leaf([1], true)?;
    let prediction = graph.binary(BinaryOp::Mul, signal, gain)?;
    let error = graph.binary(BinaryOp::Sub, prediction, target)?;
    let squared = graph.unary(UnaryOp::Square, error)?;
    let loss = graph.reduce(ReduceOp::Mean, squared, &[0], false)?;
    let program = graph.finish(&[loss])?;

    let samples = Shared::new(Tensor::from_vec([4], vec![1., -1., 1., -1.])?)?;
    let expected = Shared::new(Tensor::from_vec([4], vec![2., -2., 2., -2.])?)?;
    let weight = Shared::new(Tensor::from_vec([1], vec![0.])?)?;
    let bindings = [
        Binding::new(signal, samples),
        Binding::new(target, expected),
        Binding::new(gain, weight.clone()),
    ];
    let mut tape = Tape::new(Cpu::Reference, &program, &bindings)?;
    for _ in 0..12 {
        tape.step_scalar(&[gain], 0.25)?;
    }

    let learned = weight.read()?.data()[0];
    assert!((learned - 2.0).abs() < 0.001);
    println!("Gain: {learned:.4}");
    Ok(())
}

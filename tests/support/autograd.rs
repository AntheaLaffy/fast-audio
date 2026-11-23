use fast_audio::{
    BinaryOp as B, ReduceOp as R, Tensor, UnaryOp as U,
    autograd::{Builder, Program, Value},
};

pub enum Formula {
    Unary(U),
    Binary(B),
}
pub struct Fixture {
    pub program: Program,
    pub leaves: [Value; 2],
    pub inputs: [Tensor; 2],
    pub seed: Tensor,
    formula: Formula,
}
pub struct Reference {
    pub output: Vec<f64>,
    pub gradients: [Vec<f64>; 2],
}

fn unary(op: U, x: f64) -> (f64, f64) {
    let sigmoid = |x: f64| {
        if x >= 0. {
            1. / (1. + (-x).exp())
        } else {
            let e = x.exp();
            e / (1. + e)
        }
    };
    match op {
        U::Neg => (-x, -1.),
        U::Abs => (x.abs(), x.signum()),
        U::Square => (x * x, 2. * x),
        U::Reciprocal => (1. / x, -1. / (x * x)),
        U::Sqrt => (x.sqrt(), 0.5 / x.sqrt()),
        U::Exp => (x.exp(), x.exp()),
        U::Log => (x.ln(), 1. / x),
        U::Relu => (x.max(0.), if x > 0. { 1. } else { 0. }),
        U::Sigmoid => {
            let s = sigmoid(x);
            (s, s * (1. - s))
        }
        U::Tanh => (x.tanh(), 1. - x.tanh().powi(2)),
        U::Silu => {
            let s = sigmoid(x);
            (x * s, s + x * s * (1. - s))
        }
        U::Softplus => (x.max(0.) + (-x.abs()).exp().ln_1p(), sigmoid(x)),
    }
}
fn binary(op: B, a: f64, b: f64) -> (f64, f64, f64) {
    match op {
        B::Add => (a + b, 1., 1.),
        B::Sub => (a - b, 1., -1.),
        B::Mul => (a * b, b, a),
        B::Div => (a / b, 1. / b, -a / (b * b)),
    }
}
impl Fixture {
    /// Independent analytic equations for a transposed nonlinear branch or an
    /// asymmetric broadcast objective; no production plans or traversal are used.
    pub fn reference(&self, inputs: &[Tensor; 2]) -> Reference {
        let mut data = inputs
            .each_ref()
            .map(|t| t.data().iter().map(|&v| v as f64).collect::<Vec<_>>());
        let reference = self.evaluate(&data);
        assert_eq!(reference.output.len(), self.seed.data().len());
        let objective = |data: &[Vec<f64>; 2]| {
            self.evaluate(data)
                .output
                .iter()
                .zip(self.seed.data())
                .map(|(&v, &s)| v * s as f64)
                .sum::<f64>()
        };
        for side in 0..2 {
            for coordinate in 0..data[side].len() {
                let original = data[side][coordinate];
                data[side][coordinate] = original + 1e-4;
                let plus = objective(&data);
                data[side][coordinate] = original - 1e-4;
                let minus = objective(&data);
                data[side][coordinate] = original;
                let difference = (plus - minus) / 2e-4;
                let expected = reference.gradients[side][coordinate];
                assert!(
                    (difference - expected).abs() <= 2e-5 + 2e-5 * expected.abs(),
                    "composed analytic/finite-difference control differs"
                );
            }
        }
        reference
    }
    fn evaluate(&self, inputs: &[Vec<f64>; 2]) -> Reference {
        let mut gradients = [vec![0.; inputs[0].len()], vec![0.; inputs[1].len()]];
        let output = match self.formula {
            Formula::Unary(op) => {
                let mut output = vec![0.; 3];
                for i in 0..2 {
                    for j in 0..3 {
                        let (y, dy) = unary(op, inputs[0][i * 3 + j]);
                        output[j] += (y * y + inputs[1][j]) / 2.;
                        gradients[0][i * 3 + j] = self.seed.data()[j] as f64 * y * dy;
                        gradients[1][j] = self.seed.data()[j] as f64;
                    }
                }
                output
            }
            Formula::Binary(op) => {
                let mut total = 0.;
                for i in 0..2 {
                    for j in 0..3 {
                        let (y, da, db) = binary(op, inputs[0][i], inputs[1][j]);
                        total += y * y;
                        gradients[0][i] += 2. * y * da;
                        gradients[1][j] += 2. * y * db;
                    }
                }
                vec![total]
            }
        };
        Reference { output, gradients }
    }
    pub fn changed_inputs(&self, binding: usize) -> [Tensor; 2] {
        let mut result = self.inputs.clone();
        if binding == 1 {
            for t in &mut result {
                for v in t.data_mut() {
                    *v += 0.07;
                }
            }
        }
        result
    }
}
pub fn fixtures() -> Vec<Fixture> {
    let mut cases = Vec::new();
    for op in [
        U::Neg,
        U::Abs,
        U::Square,
        U::Reciprocal,
        U::Sqrt,
        U::Exp,
        U::Log,
        U::Relu,
        U::Sigmoid,
        U::Tanh,
        U::Silu,
        U::Softplus,
    ] {
        let mut b = Builder::new().unwrap();
        let x = b.leaf([2, 3], true).unwrap();
        let bias = b.leaf([3], true).unwrap();
        let y = b.unary(op, x).unwrap();
        let view = b.permute(y, &[1, 0]).unwrap();
        let squared = b.binary(B::Mul, view, view).unwrap();
        let bias_view = b.reshape(bias, [3, 1]).unwrap();
        let shifted = b.binary(B::Add, squared, bias_view).unwrap();
        let root = b.reduce(R::Mean, shifted, &[1], false).unwrap();
        let input = if matches!(op, U::Sqrt | U::Log) {
            vec![0.4, 0.8, 1.2, 1.6, 2., 2.4]
        } else {
            vec![-0.8, -0.3, 0.2, 0.7, 1.2, 1.7]
        };
        cases.push(Fixture {
            program: b.finish(&[root]).unwrap(),
            leaves: [x, bias],
            inputs: [
                Tensor::from_vec([2, 3], input).unwrap(),
                Tensor::from_vec([3], vec![0.1, 0.2, 0.3]).unwrap(),
            ],
            seed: Tensor::from_vec([3], vec![0.5, -0.75, 1.25]).unwrap(),
            formula: Formula::Unary(op),
        });
    }
    for op in [B::Add, B::Sub, B::Mul, B::Div] {
        let mut b = Builder::new().unwrap();
        let a = b.leaf([2], true).unwrap();
        let right = b.leaf([3], true).unwrap();
        let left = b.reshape(a, [2, 1]).unwrap();
        let y = b.binary(op, left, right).unwrap();
        let square = b.unary(U::Square, y).unwrap();
        let root = b.reduce(R::Sum, square, &[0, 1], false).unwrap();
        cases.push(Fixture {
            program: b.finish(&[root]).unwrap(),
            leaves: [a, right],
            inputs: [
                Tensor::from_vec([2], vec![-0.4, 0.9]).unwrap(),
                Tensor::from_vec([3], vec![0.5, 1.3, 2.1]).unwrap(),
            ],
            seed: Tensor::from_vec([], vec![1.]).unwrap(),
            formula: Formula::Binary(op),
        });
    }
    cases
}

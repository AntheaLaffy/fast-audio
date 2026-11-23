use fast_audio::{
    BinaryOp as B, ReduceOp as R, Tensor, UnaryOp as U,
    autograd::{Builder, Program, Value},
};

/// Procedural tensor objectives, with no task model or FFT derivative implied.
pub struct Fixture {
    pub name: &'static str,
    pub program: Program,
    /// Input, target, channel/bin gain, and channel/component bias.
    pub leaves: [Value; 4],
    pub input: Tensor,
    pub target: Tensor,
    pub parameters: [Tensor; 2],
}
pub struct Reference {
    pub prediction: Vec<f64>,
    pub loss: f64,
    pub gradients: [Vec<f64>; 2],
}
impl Fixture {
    pub fn initial(&self) -> [Vec<f64>; 2] {
        self.parameters
            .each_ref()
            .map(|t| t.data().iter().map(|&v| v as f64).collect())
    }
    /// Scalar equations and physical parameter indexing are independent of the DAG.
    pub fn reference(&self, parameters: &[Vec<f64>; 2]) -> Reference {
        let mut prediction = vec![0.; self.target.shape().len()];
        let mut gradients = parameters.each_ref().map(|p| vec![0.; p.len()]);
        let mut loss = 0.;
        let count = prediction.len() as f64;
        if self.name == "waveform" {
            for batch in 0..2 {
                for time in 0..17 {
                    for channel in 0..2 {
                        let x = self.input.data()[(batch * 2 + channel) * 17 + 16 - time] as f64;
                        let y = (x * parameters[0][channel] + parameters[1][channel]).tanh();
                        let i = (batch * 17 + time) * 2 + channel;
                        prediction[i] = y;
                        let residual = y - self.target.data()[i] as f64;
                        loss += residual * residual / count;
                        let dy = 2. * residual / count * (1. - y * y);
                        gradients[0][channel] += dy * x;
                        gradients[1][channel] += dy;
                    }
                }
            }
        } else {
            assert_eq!(self.name, "spectrum");
            for batch in 0..2 {
                for time in 0..5 {
                    for channel in 0..2 {
                        for bin in 0..3 {
                            let offset = (((batch * 2 + channel) * 3 + bin) * 5 + 4 - time) * 2;
                            let xs = self.input.data()[offset..offset + 2]
                                .iter()
                                .map(|&v| v as f64)
                                .collect::<Vec<_>>();
                            let zs = [
                                xs[0] * parameters[0][channel * 3 + bin] + parameters[1][0],
                                xs[1] * parameters[0][channel * 3 + bin] + parameters[1][1],
                            ];
                            let y = zs.iter().map(|v| v * v).sum::<f64>();
                            let i = ((batch * 5 + time) * 2 + channel) * 3 + bin;
                            prediction[i] = y;
                            let residual = y - self.target.data()[i] as f64;
                            loss += residual * residual / count;
                            for component in 0..2 {
                                let dz = 4. * residual * zs[component] / count;
                                gradients[0][channel * 3 + bin] += dz * xs[component];
                                gradients[1][component] += dz;
                            }
                        }
                    }
                }
            }
        }
        Reference {
            prediction,
            loss,
            gradients,
        }
    }
    pub fn check_finite_differences(&self) {
        let mut parameters = self.initial();
        let analytic = self.reference(&parameters);
        for side in 0..2 {
            for coordinate in 0..parameters[side].len() {
                let original = parameters[side][coordinate];
                parameters[side][coordinate] = original + 1e-4;
                let plus = self.reference(&parameters).loss;
                parameters[side][coordinate] = original - 1e-4;
                let minus = self.reference(&parameters).loss;
                parameters[side][coordinate] = original;
                let expected = analytic.gradients[side][coordinate];
                assert!(((plus - minus) / 2e-4 - expected).abs() <= 2e-5 + 2e-5 * expected.abs());
            }
        }
    }
}
pub fn fixtures() -> Vec<Fixture> {
    let mut result = Vec::new();
    for name in ["waveform", "spectrum"] {
        let waveform = name == "waveform";
        let input_dims: &[usize] = if waveform {
            &[2, 2, 17]
        } else {
            &[2, 2, 3, 5, 2]
        };
        let target_dims: &[usize] = if waveform { &[2, 17, 2] } else { &[2, 5, 2, 3] };
        let gain_dims: &[usize] = if waveform { &[2] } else { &[2, 3] };
        let mut b = Builder::new().unwrap();
        let x = b.leaf(input_dims, false).unwrap();
        let target_value = b.leaf(target_dims, false).unwrap();
        let gain = b.leaf(gain_dims, true).unwrap();
        let bias = b.leaf([2], true).unwrap();
        let gain_view = b
            .reshape(
                gain,
                if waveform {
                    vec![1, 2, 1]
                } else {
                    vec![1, 2, 3, 1, 1]
                },
            )
            .unwrap();
        let bias_view = b
            .reshape(
                bias,
                if waveform {
                    vec![1, 2, 1]
                } else {
                    vec![1, 1, 1, 1, 2]
                },
            )
            .unwrap();
        let scaled = b.binary(B::Mul, x, gain_view).unwrap();
        let shifted = b.binary(B::Add, scaled, bias_view).unwrap();
        let prediction = if waveform {
            let bounded = b.unary(U::Tanh, shifted).unwrap();
            let reversed = b.slice(bounded, 2, 16, 17, -1).unwrap();
            b.permute(reversed, &[0, 2, 1]).unwrap()
        } else {
            let square = b.unary(U::Square, shifted).unwrap();
            let power = b.reduce(R::Sum, square, &[4], false).unwrap();
            let reversed = b.slice(power, 3, 4, 5, -1).unwrap();
            b.permute(reversed, &[0, 3, 1, 2]).unwrap()
        };
        let residual = b.binary(B::Sub, prediction, target_value).unwrap();
        let square = b.unary(U::Square, residual).unwrap();
        let axes = (0..target_dims.len()).collect::<Vec<_>>();
        let loss = b.reduce(R::Mean, square, &axes, false).unwrap();
        let program = b.finish(&[loss, prediction]).unwrap();
        let input_len = input_dims.iter().product();
        let input = Tensor::from_vec(
            input_dims,
            (0..input_len)
                .map(|i| ((i * 7 + 3) % 29) as f32 / 32. - 0.5)
                .collect(),
        )
        .unwrap();
        let target_len = target_dims.iter().product();
        let target = Tensor::from_vec(
            target_dims,
            (0..target_len)
                .map(|i| {
                    if waveform {
                        ((i * 5 + 1) % 23) as f32 / 32. - 0.375
                    } else {
                        ((i * 3 + 2) % 17) as f32 / 64. + 0.0625
                    }
                })
                .collect(),
        )
        .unwrap();
        let gain_len = gain_dims.iter().product();
        let parameters = [
            Tensor::from_vec(
                gain_dims,
                (0..gain_len).map(|i| 0.5 + i as f32 / 16.).collect(),
            )
            .unwrap(),
            Tensor::from_vec([2], vec![-0.0625, 0.125]).unwrap(),
        ];
        let fixture = Fixture {
            name,
            program,
            leaves: [x, target_value, gain, bias],
            input,
            target,
            parameters,
        };
        fixture.check_finite_differences();
        result.push(fixture);
    }
    result
}

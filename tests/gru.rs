use fast_audio::{
    GruSpec, GruWeights, Tensor,
    cpu::{Cpu, Gru},
};
mod support;
use support::{backends, close, data, no_allocations};

fn cases() -> Vec<GruSpec> {
    [
        (1, 1, 3, 5, false),
        (2, 7, 5, 3, false),
        (2, 4, 3, 7, true),
        (3, 33, 8, 4, true),
        (33, 17, 8, 8, false),
        (1, 63, 8, 16, false),
        (1, 2, 3, 193, false),
    ]
    .into_iter()
    .map(
        |(batch, steps, input_size, hidden_size, bidirectional)| GruSpec {
            batch,
            steps,
            input_size,
            hidden_size,
            bidirectional,
        },
    )
    .collect()
}
fn weights(s: GruSpec) -> GruWeights {
    let d = s.directions();
    let g = s.hidden_size * 3;
    let mut p = GruWeights {
        input: data([d, g, s.input_size]),
        hidden: data([d, g, s.hidden_size]),
        input_bias: data([d, g]),
        hidden_bias: data([d, g]),
    };
    for tensor in [&mut p.input, &mut p.hidden] {
        for value in tensor.data_mut() {
            *value *= 0.3;
        }
    }
    for value in p.input_bias.data_mut() {
        *value *= 0.7;
    }
    for value in p.hidden_bias.data_mut() {
        *value *= 0.4;
    }
    p
}
fn oracle(s: GruSpec, p: &GruWeights, x: &Tensor, initial: &Tensor) -> (Vec<f64>, Vec<f64>) {
    let d = s.directions();
    let h = s.hidden_size;
    let mut output = vec![0.0; s.batch * s.steps * d * h];
    let mut final_state = Vec::new();
    for direction in 0..d {
        let mut state: Vec<_> = initial.data()[direction * s.batch * h..][..s.batch * h]
            .iter()
            .map(|&v| v as f64)
            .collect();
        let times: Vec<_> = if direction == 0 {
            (0..s.steps).collect()
        } else {
            (0..s.steps).rev().collect()
        };
        for time in times {
            let old = state.clone();
            for batch in 0..s.batch {
                for unit in 0..h {
                    let mut gates = [0.0; 3];
                    let mut reset = 0.0;
                    for (gate, value) in gates.iter_mut().enumerate() {
                        let row = (direction * 3 + gate) * h + unit;
                        let input_sum = p.input_bias.data()[row] as f64
                            + (0..s.input_size)
                                .map(|i| {
                                    x.data()[(batch * s.steps + time) * s.input_size + i] as f64
                                        * p.input.data()[row * s.input_size + i] as f64
                                })
                                .sum::<f64>();
                        let hidden_sum = p.hidden_bias.data()[row] as f64
                            + (0..h)
                                .map(|i| old[batch * h + i] * p.hidden.data()[row * h + i] as f64)
                                .sum::<f64>();
                        // The candidate gate's recurrent bias is inside reset multiplication.
                        *value = if gate < 2 {
                            1.0 / (1.0 + (-(input_sum + hidden_sum)).exp())
                        } else {
                            (input_sum + reset * hidden_sum).tanh()
                        };
                        if gate == 0 {
                            reset = *value;
                        }
                    }
                    let value = (1.0 - gates[1]) * gates[2] + gates[1] * old[batch * h + unit];
                    state[batch * h + unit] = value;
                    output[((batch * s.steps + time) * d + direction) * h + unit] = value;
                }
            }
        }
        final_state.extend(state);
    }
    (output, final_state)
}

#[test]
fn cpu_gru_matches_f64_for_direction_state_and_repeated_execution() {
    for cpu in backends() {
        for s in cases() {
            let p = weights(s);
            let mut x = data(s.input_shape());
            let initial = data(s.state_shape());
            let mut y = Tensor::zeros(s.output_shape().unwrap()).unwrap();
            let mut final_state = Tensor::zeros(s.state_shape()).unwrap();
            let mut plan = Gru::new(cpu, s, &p).unwrap();
            assert!(plan.workspace_bytes() > 0 && plan.parameter_bytes() > 0);
            for _ in 0..2 {
                let expected = oracle(s, &p, &x, &initial);
                plan.run(&x, &initial, &mut y, &mut final_state).unwrap();
                close(y.data(), &expected.0, 2e-6, 2e-5);
                close(final_state.data(), &expected.1, 2e-6, 2e-5);
                no_allocations(|| plan.run(&x, &initial, &mut y, &mut final_state).unwrap());
                for value in x.data_mut() {
                    *value = 0.17 - *value;
                }
            }
        }
    }
}

#[test]
fn cpu_gru_snapshots_parameters_and_rejects_wrong_axes() {
    let s = cases()[2];
    let mut p = weights(s);
    let x = data(s.input_shape());
    let initial = data(s.state_shape());
    let expected = oracle(s, &p, &x, &initial);
    let mut plan = Gru::new(Cpu::Reference, s, &p).unwrap();
    p.input.data_mut().fill(99.0);
    p.hidden_bias.data_mut().fill(-99.0);
    let mut y = Tensor::zeros(s.output_shape().unwrap()).unwrap();
    let mut final_state = Tensor::zeros(s.state_shape()).unwrap();
    plan.run(&x, &initial, &mut y, &mut final_state).unwrap();
    close(y.data(), &expected.0, 2e-6, 2e-5);
    let wrong = data([s.steps, s.batch, s.input_size]);
    assert!(
        plan.run(&wrong, &initial, &mut y, &mut final_state)
            .is_err()
    );
    let wrong = data([s.batch, s.directions(), s.hidden_size]);
    // This case has batch == directions; use a different rank with equal element count.
    let wrong = Tensor::from_vec([wrong.shape().len()], wrong.into_vec()).unwrap();
    assert!(plan.run(&x, &wrong, &mut y, &mut final_state).is_err());
    let mut invalid = weights(s);
    invalid.hidden = Tensor::zeros([1, 1, 1]).unwrap();
    assert!(Gru::new(Cpu::Reference, s, &invalid).is_err());
    for bad in [
        GruSpec { steps: 0, ..s },
        GruSpec { input_size: 0, ..s },
        GruSpec {
            hidden_size: usize::MAX,
            ..s
        },
        GruSpec {
            batch: i32::MAX as usize,
            ..s
        },
    ] {
        assert!(bad.output_shape().is_err());
    }
}

#[test]
fn forward_gru_state_connects_chunks_without_hidden_plan_state() {
    let s = GruSpec {
        steps: 6,
        ..cases()[1]
    };
    let p = weights(s);
    let x = data(s.input_shape());
    let initial = data(s.state_shape());
    let mut expected = Tensor::zeros(s.output_shape().unwrap()).unwrap();
    let mut final_expected = Tensor::zeros(s.state_shape()).unwrap();
    Gru::new(Cpu::Reference, s, &p)
        .unwrap()
        .run(&x, &initial, &mut expected, &mut final_expected)
        .unwrap();
    let part = GruSpec { steps: 3, ..s };
    let mut plan = Gru::new(Cpu::Reference, part, &p).unwrap();
    let mut state = initial;
    let mut stitched = vec![0.0; expected.data().len()];
    for chunk in 0..2 {
        let mut input = Tensor::zeros(part.input_shape()).unwrap();
        for b in 0..s.batch {
            input.data_mut()[b * 3 * s.input_size..][..3 * s.input_size].copy_from_slice(
                &x.data()[(b * s.steps + chunk * 3) * s.input_size..][..3 * s.input_size],
            );
        }
        let mut y = Tensor::zeros(part.output_shape().unwrap()).unwrap();
        let mut next = Tensor::zeros(part.state_shape()).unwrap();
        plan.run(&input, &state, &mut y, &mut next).unwrap();
        for b in 0..s.batch {
            stitched[(b * s.steps + chunk * 3) * s.hidden_size..][..3 * s.hidden_size]
                .copy_from_slice(&y.data()[b * 3 * s.hidden_size..][..3 * s.hidden_size]);
        }
        state = next;
    }
    assert_eq!(stitched, expected.data());
    assert_eq!(state.data(), final_expected.data());
}

#[cfg(feature = "cudnn")]
#[test]
fn cudnn_gru_matches_independent_oracle_and_preserves_inputs() {
    use fast_audio::cuda;
    let gpu = cuda::Cuda::new(0).unwrap();
    for s in cases() {
        let mut p = weights(s);
        let x = data(s.input_shape());
        let initial = data(s.state_shape());
        let expected = oracle(s, &p, &x, &initial);
        let mut plan = cuda::Gru::new(&gpu, s, &p).unwrap();
        p.hidden.data_mut().fill(-88.0);
        let x = gpu.upload(&x).unwrap();
        let initial = gpu.upload(&initial).unwrap();
        let mut y = gpu.zeros(s.output_shape().unwrap()).unwrap();
        let mut final_state = gpu.zeros(s.state_shape()).unwrap();
        let saved_x = gpu.download(&x).unwrap();
        let saved_initial = gpu.download(&initial).unwrap();
        for _ in 0..2 {
            plan.run(&x, &initial, &mut y, &mut final_state).unwrap();
        }
        close(gpu.download(&y).unwrap().data(), &expected.0, 2e-6, 2e-5);
        close(
            gpu.download(&final_state).unwrap().data(),
            &expected.1,
            2e-6,
            2e-5,
        );
        assert_eq!(gpu.download(&x).unwrap().data(), saved_x.data());
        assert_eq!(gpu.download(&initial).unwrap().data(), saved_initial.data());
        no_allocations(|| plan.run(&x, &initial, &mut y, &mut final_state).unwrap());
        let other = cuda::Cuda::new(0).unwrap();
        let foreign = other.zeros(s.state_shape()).unwrap();
        assert!(plan.run(&x, &foreign, &mut y, &mut final_state).is_err());
    }
}

#[cfg(feature = "cudnn")]
#[test]
fn cudnn_gru_exposes_standard_control_and_rejects_vendor_sequence_limit() {
    use fast_audio::cuda::{Cuda, Gru, GruAlgorithm, GruLayout, GruOptions};
    let gpu = Cuda::new(0).unwrap();
    let s = cases()[3];
    let p = weights(s);
    let x = data(s.input_shape());
    let initial = data(s.state_shape());
    let expected = oracle(s, &p, &x, &initial);
    let x = gpu.upload(&x).unwrap();
    let initial = gpu.upload(&initial).unwrap();
    let mut y = gpu.zeros(s.output_shape().unwrap()).unwrap();
    let mut state = gpu.zeros(s.state_shape()).unwrap();
    for options in [
        GruOptions {
            algorithm: GruAlgorithm::Standard,
            layout: GruLayout::BatchMajor,
        },
        GruOptions {
            algorithm: GruAlgorithm::Automatic,
            layout: GruLayout::BatchMajor,
        },
    ] {
        let mut plan = Gru::with_options(&gpu, s, &p, options).unwrap();
        assert!(plan.implementation().contains("batch-major"));
        plan.run(&x, &initial, &mut y, &mut state).unwrap();
        close(gpu.download(&y).unwrap().data(), &expected.0, 2e-6, 2e-5);
        close(
            gpu.download(&state).unwrap().data(),
            &expected.1,
            2e-6,
            2e-5,
        );
    }
    assert!(Gru::new(&gpu, GruSpec { steps: 65536, ..s }, &p).is_err());
}

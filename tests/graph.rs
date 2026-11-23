use fast_audio::{
    GruSpec, GruWeights, Tensor,
    cpu::{Cpu, Graph},
    graph::{Builder, Program, Unary},
};
mod support;
use support::{backends, close, data, no_allocations};

fn case(which: usize) -> (Tensor, Program, Vec<f64>) {
    let x = if (13..=14).contains(&which) {
        data([2, 3, 5, 2])
    } else {
        data([2, 3, 4, 5])
    };
    let mut g = Builder::new(x.shape().dims()).unwrap();
    let root = g.input();
    let mut expected = Vec::new();
    let at =
        |b: usize, c: usize, t: usize, f: usize| x.data()[((b * 3 + c) * 4 + t) * 5 + f] as f64;
    let y = match which {
        0 => {
            for b in 0..2 {
                for t in 1..3 {
                    for f in 1..4 {
                        for c in 1..3 {
                            expected.push(at(b, c, t, f));
                        }
                    }
                }
            }
            g.layout(
                root,
                &[(0, 2), (1, 2), (1, 2), (1, 3)],
                &[0, 2, 3, 1],
                [4, 3, 2],
            )
            .unwrap()
        }
        1..=3 => {
            let operation = match which {
                1 => Unary::Sigmoid,
                2 => Unary::Tanh,
                _ => Unary::PRelu(0.17),
            };
            expected = x
                .data()
                .iter()
                .map(|&v| {
                    let v = v as f64;
                    match which {
                        1 => 1.0 / (1.0 + (-v).exp()),
                        2 => v.tanh(),
                        _ => {
                            if v < 0.0 {
                                v * 0.17f32 as f64
                            } else {
                                v
                            }
                        }
                    }
                })
                .collect();
            g.unary(root, operation).unwrap()
        }
        4 => {
            expected = x.data().iter().map(|&v| 2.0 * v as f64).collect();
            g.add(root, root).unwrap()
        }
        5 => {
            let multipliers = [0.5, -0.25, 2.0];
            expected = x
                .data()
                .iter()
                .enumerate()
                .map(|(i, &v)| v as f64 * multipliers[i / 20 % 3] as f64)
                .collect();
            let rhs = g
                .constant(Tensor::from_vec([3, 1, 1], multipliers.to_vec()).unwrap())
                .unwrap();
            g.multiply_broadcast(root, rhs).unwrap()
        }
        6 => {
            for b in 0..2 {
                for c in 0..3 {
                    for branch in 0..2 {
                        for t in 0..4 {
                            for f in 0..5 {
                                let v = at(b, c, t, f);
                                expected.push(if branch == 0 { v } else { v.tanh() });
                            }
                        }
                    }
                }
            }
            let rhs = g.unary(root, Unary::Tanh).unwrap();
            g.concat(root, rhs, 1, true).unwrap()
        }
        7 => {
            for b in 0..2 {
                for c in 0..3 {
                    for t in 0..7 {
                        for f in 0..5 {
                            expected.push(if (2..6).contains(&t) {
                                at(b, c, t - 2, f)
                            } else {
                                0.0
                            });
                        }
                    }
                }
            }
            g.pad_axis(root, 2, 2, 1).unwrap()
        }
        8 | 15 => {
            let width = if which == 8 { 3 } else { 13 };
            for b in 0..2 {
                for c in 0..3 {
                    for k in 0..width {
                        for t in 0..4 {
                            for f in 0..5 {
                                let q = f as isize + k as isize - (width / 2) as isize;
                                expected.push(if (0..5).contains(&q) {
                                    at(b, c, t, q as usize)
                                } else {
                                    0.0
                                });
                            }
                        }
                    }
                }
            }
            g.unfold_frequency(root, width).unwrap()
        }
        9 => {
            expected = x
                .data()
                .chunks_exact(5)
                .map(|row| row.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / 5.0)
                .collect();
            g.mean_square_last(root).unwrap()
        }
        10 => {
            let scale = data([4, 5]);
            let bias = data([4, 5]);
            for row in x.data().chunks_exact(20) {
                let mean = row.iter().map(|&v| v as f64).sum::<f64>() / 20.0;
                let variance = row.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / 20.0;
                for (i, &v) in row.iter().enumerate() {
                    expected.push(
                        (v as f64 - mean) / (variance + 1e-3f32 as f64).sqrt()
                            * scale.data()[i] as f64
                            + bias.data()[i] as f64,
                    );
                }
            }
            g.layer_norm(root, &scale, &bias, 1e-3).unwrap()
        }
        11 => {
            let mut w = data([7, 5]);
            let mut bias = data([7]);
            for row in x.data().chunks_exact(5) {
                for o in 0..7 {
                    expected.push(
                        bias.data()[o] as f64
                            + (0..5)
                                .map(|i| row[i] as f64 * w.data()[o * 5 + i] as f64)
                                .sum::<f64>(),
                    );
                }
            }
            let result = g.linear(root, &w, Some(&bias), [2, 3, 4, 7]).unwrap();
            w.data_mut().fill(100.0);
            bias.data_mut().fill(-50.0);
            result
        }
        12 => {
            expected = x.data().iter().map(|&v| v as f64).collect();
            let a = g
                .layout(
                    root,
                    &[(0, 2), (0, 1), (0, 4), (0, 5)],
                    &[0, 1, 2, 3],
                    [2, 1, 4, 5],
                )
                .unwrap();
            let b = g
                .layout(
                    root,
                    &[(0, 2), (1, 2), (0, 4), (0, 5)],
                    &[0, 1, 2, 3],
                    [2, 2, 4, 5],
                )
                .unwrap();
            g.concat(a, b, 1, false).unwrap()
        }
        13 => {
            for b in 0..2 {
                for c in 0..3 {
                    for p in 0..15 {
                        let re = x.data()[(b * 15 + p) * 2] as f64;
                        let im = x.data()[(b * 15 + p) * 2 + 1] as f64;
                        expected.push(match c {
                            0 => (re * re + im * im + 1e-12f32 as f64).sqrt(),
                            1 => re,
                            _ => im,
                        });
                    }
                }
            }
            g.complex_features(root, 1e-12).unwrap()
        }
        14 => {
            let mask = data([2, 2, 3, 5]);
            for b in 0..2 {
                for p in 0..15 {
                    let re = x.data()[(b * 15 + p) * 2] as f64;
                    let im = x.data()[(b * 15 + p) * 2 + 1] as f64;
                    let r = mask.data()[b * 30 + p] as f64;
                    let i = mask.data()[b * 30 + 15 + p] as f64;
                    expected.extend([re * r - im * i, im * r + re * i]);
                }
            }
            let mask = g.constant(mask).unwrap();
            g.complex_mask(mask, root).unwrap()
        }
        _ => unreachable!(),
    };
    (x, g.finish(y).unwrap(), expected)
}
#[test]
fn cpu_graph_kernels_match_independent_f64_and_reuse_storage() {
    for cpu in backends() {
        for which in 0..16 {
            let (x, program, expected) = case(which);
            let saved = x.data().to_vec();
            let mut graph = Graph::new(cpu, program).unwrap();
            let mut out = Tensor::zeros(graph.output_shape()).unwrap();
            for _ in 0..2 {
                graph.run(&x, &mut out).unwrap();
            }
            close(out.data(), &expected, 3e-6, 3e-5);
            no_allocations(|| graph.run(&x, &mut out).unwrap());
            assert_eq!(x.data(), saved);
        }
    }
}
fn recurrent() -> (Program, fast_audio::graph::Value) {
    let mut b = Builder::new([1, 3, 1]).unwrap();
    let input = b.input();
    let initial = b
        .constant(Tensor::from_vec([1, 1, 1], vec![0.75]).unwrap())
        .unwrap();
    let weights = GruWeights {
        input: Tensor::zeros([1, 3, 1]).unwrap(),
        hidden: Tensor::zeros([1, 3, 1]).unwrap(),
        input_bias: Tensor::zeros([1, 3]).unwrap(),
        hidden_bias: Tensor::zeros([1, 3]).unwrap(),
    };
    let (y, h) = b
        .gru(
            input,
            initial,
            GruSpec {
                batch: 1,
                steps: 3,
                input_size: 1,
                hidden_size: 1,
                bidirectional: false,
            },
            weights,
        )
        .unwrap();
    let out = b.add(y, y).unwrap();
    (b.finish(out).unwrap(), h)
}
#[test]
fn graph_recurrence_keeps_initial_and_final_state_distinct_across_calls() {
    for cpu in backends() {
        let (program, h) = recurrent();
        let mut g = Graph::new(cpu, program).unwrap();
        let input = data([1, 3, 1]);
        let mut out = Tensor::zeros([1, 3, 1]).unwrap();
        for _ in 0..3 {
            g.run(&input, &mut out).unwrap();
            assert_eq!(out.data(), [0.75, 0.375, 0.1875]);
            assert_eq!(g.intermediate(h).unwrap().data(), [0.09375]);
        }
        no_allocations(|| g.run(&input, &mut out).unwrap());
    }
}
#[test]
fn graphs_reject_foreign_values_bad_geometry_and_equal_size_wrong_axes() {
    let mut g = Builder::new([2, 3, 4]).unwrap();
    let x = g.input();
    let foreign = Builder::new([2, 3, 4]).unwrap().input();
    assert!(g.add(x, foreign).is_err());
    assert!(
        g.layout(x, &[(0, 2), (0, 3), (0, 4)], &[0, 0, 2], [2, 3, 4])
            .is_err()
    );
    assert!(
        g.layout(x, &[(0, 2), (0, 3), (3, 2)], &[0, 1, 2], [2, 3, 2])
            .is_err()
    );
    assert!(g.unary(x, Unary::PRelu(f32::NAN)).is_err());
    assert!(g.layer_norm(x, &data([3]), &data([3]), 1e-5).is_err());
    assert!(g.layer_norm(x, &data([4]), &data([4]), 0.0).is_err());
    assert!(g.complex_features(x, 1e-12).is_err());
    let y = g.unary(x, Unary::Tanh).unwrap();
    let mut graph = Graph::new(Cpu::Reference, g.finish(y).unwrap()).unwrap();
    assert!(graph.intermediate(foreign).is_none());
    let mut out = Tensor::zeros([2, 3, 4]).unwrap();
    assert!(graph.run(&data([6, 4]), &mut out).is_err());
    assert!(
        graph
            .run(&data([2, 3, 4]), &mut Tensor::zeros([6, 4]).unwrap())
            .is_err()
    );
}
#[cfg(feature = "cudnn")]
#[test]
fn cuda_graph_kernels_match_f64_and_reject_foreign_sessions() {
    use fast_audio::cuda;
    let gpu = cuda::Cuda::new(0).unwrap();
    let other = cuda::Cuda::new(0).unwrap();
    for which in 0..16 {
        let (x, program, expected) = case(which);
        let input = gpu.upload(&x).unwrap();
        let mut graph = cuda::Graph::new(&gpu, program).unwrap();
        let mut out = gpu.zeros(graph.output_shape()).unwrap();
        graph.run(&input, &mut out).unwrap();
        close(gpu.download(&out).unwrap().data(), &expected, 3e-6, 3e-5);
        no_allocations(|| graph.run(&input, &mut out).unwrap());
        assert!(graph.run(&other.upload(&x).unwrap(), &mut out).is_err());
        assert!(
            graph
                .run(&input, &mut other.zeros(graph.output_shape()).unwrap())
                .is_err()
        );
        let replacement = gpu.upload(&x).unwrap();
        let mut replacement_out = gpu.zeros(graph.output_shape()).unwrap();
        graph.run(&replacement, &mut replacement_out).unwrap();
        close(
            gpu.download(&replacement_out).unwrap().data(),
            &expected,
            3e-6,
            3e-5,
        );
    }
    let (program, h) = recurrent();
    let mut graph = cuda::Graph::new(&gpu, program).unwrap();
    let x = gpu.upload(&data([1, 3, 1])).unwrap();
    let mut y = gpu.zeros([1, 3, 1]).unwrap();
    for _ in 0..3 {
        graph.run(&x, &mut y).unwrap();
        close(
            gpu.download(&y).unwrap().data(),
            &[0.75, 0.375, 0.1875],
            1e-6,
            1e-6,
        );
        close(
            gpu.download(graph.intermediate(h).unwrap()).unwrap().data(),
            &[0.09375],
            1e-6,
            1e-6,
        );
    }
}

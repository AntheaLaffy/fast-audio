use fast_audio::{
    Activation, CenteredStftSpec, Conv1dSpec, Shape, StftSpec, Tensor,
    cpu::{CenteredStft, Conv1d, Cpu, Istft, Stft},
};
mod support;
use support::{backends, close, data, no_allocations};

fn conv_oracle(
    s: Conv1dSpec,
    x: &Tensor,
    w: &Tensor,
    bias: Option<&Tensor>,
    relu: bool,
) -> Vec<f64> {
    let n = s.output_len().unwrap();
    let cin = s.in_channels / s.groups;
    let cout = s.out_channels / s.groups;
    let mut out = vec![0.0; s.batch * s.out_channels * n];
    for b in 0..s.batch {
        for oc in 0..s.out_channels {
            for t in 0..n {
                let mut value = bias.map_or(0.0, |bias| bias.data()[oc] as f64);
                for ic in 0..cin {
                    for k in 0..s.kernel {
                        let pos = (t * s.stride + k * s.dilation) as isize - s.padding as isize;
                        if pos >= 0 && pos < s.input_len as isize {
                            value += x.data()[(b * s.in_channels + (oc / cout) * cin + ic)
                                * s.input_len
                                + pos as usize] as f64
                                * w.data()[(oc * cin + ic) * s.kernel + k] as f64;
                        }
                    }
                }
                out[(b * s.out_channels + oc) * n + t] = if relu { value.max(0.0) } else { value };
            }
        }
    }
    out
}
fn conv_cases() -> Vec<Conv1dSpec> {
    let base = Conv1dSpec {
        batch: 2,
        in_channels: 4,
        out_channels: 6,
        input_len: 17,
        kernel: 3,
        stride: 1,
        padding: 1,
        dilation: 1,
        groups: 1,
    };
    vec![
        base,
        Conv1dSpec {
            groups: 2,
            stride: 2,
            dilation: 2,
            padding: 3,
            ..base
        },
        Conv1dSpec {
            out_channels: 4,
            groups: 4,
            ..base
        },
        Conv1dSpec {
            kernel: 1,
            padding: 0,
            groups: 2,
            ..base
        },
        Conv1dSpec {
            input_len: 1,
            kernel: 1,
            padding: 0,
            ..base
        },
    ]
}
fn stft_oracle(s: StftSpec, x: &Tensor) -> Vec<f64> {
    let mut out = Vec::new();
    for b in 0..s.batch {
        for frame in 0..s.frames().unwrap() {
            for bin in 0..s.bins() {
                let (mut re, mut im) = (0.0, 0.0);
                for t in 0..s.fft_size {
                    let window = (0.5
                        - 0.5 * (std::f64::consts::TAU * t as f64 / s.fft_size as f64).cos())
                        as f32;
                    let v = (x.data()[b * s.input_len + frame * s.hop + t] * window) as f64;
                    let angle = -std::f64::consts::TAU * (bin * t) as f64 / s.fft_size as f64;
                    re += v * angle.cos();
                    im += v * angle.sin();
                }
                out.extend_from_slice(&[re, im]);
            }
        }
    }
    out
}

fn centered_cases() -> Vec<CenteredStftSpec> {
    [2, 3, 7, 8, 15, 32]
        .into_iter()
        .flat_map(|fft_size| {
            [fft_size / 2 + 1, 3 * fft_size, 3 * fft_size + 1].map(|input_len| {
                StftSpec {
                    batch: 2,
                    input_len,
                    fft_size,
                    hop: (fft_size / 3).max(1),
                }
                .centered()
                .unwrap()
            })
        })
        .collect()
}

fn centered_oracle(spec: CenteredStftSpec, x: &Tensor) -> Vec<f64> {
    let s = spec.spec();
    let pad = s.fft_size / 2;
    let mut out = Vec::new();
    for row in x.data().chunks_exact(s.input_len) {
        // Materialize the reflected signal independently of the execution index formula.
        let padded: Vec<_> = row[1..=pad]
            .iter()
            .rev()
            .chain(row.iter())
            .chain(row[s.input_len - 1 - pad..s.input_len - 1].iter().rev())
            .copied()
            .collect();
        for frame in 0..spec.frames() {
            for bin in 0..s.bins() {
                let (mut re, mut im) = (0.0, 0.0);
                for t in 0..s.fft_size {
                    let window = (0.5
                        - 0.5 * (std::f64::consts::TAU * t as f64 / s.fft_size as f64).cos())
                        as f32;
                    let value = (padded[frame * s.hop + t] * window) as f64;
                    let angle = -std::f64::consts::TAU * (bin * t) as f64 / s.fft_size as f64;
                    re += value * angle.cos();
                    im += value * angle.sin();
                }
                out.extend([re, im]);
            }
        }
    }
    out
}

fn inverse_oracle(spec: CenteredStftSpec, spectrum: &Tensor) -> Vec<f64> {
    let s = spec.spec();
    let pad = s.fft_size / 2;
    let total = (spec.frames() - 1) * s.hop + s.fft_size;
    let window: Vec<_> = (0..s.fft_size)
        .map(|t| {
            (0.5 - 0.5 * (std::f64::consts::TAU * t as f64 / s.fft_size as f64).cos()) as f32 as f64
        })
        .collect();
    let mut output = Vec::new();
    for b in 0..s.batch {
        let mut signal = vec![0.0; total];
        let mut envelope = vec![0.0; total];
        for frame in 0..spec.frames() {
            let frequency =
                &spectrum.data()[(b * spec.frames() + frame) * s.bins() * 2..][..s.bins() * 2];
            for (t, &w) in window.iter().enumerate() {
                let mut value = 0.0;
                // Full FP64 inverse DFT, including the redundant conjugate frequencies.
                for bin in 0..s.fft_size {
                    let k = if bin < s.bins() {
                        bin
                    } else {
                        s.fft_size - bin
                    };
                    let re = frequency[2 * k] as f64;
                    let im = if k == 0 || (s.fft_size % 2 == 0 && k == s.fft_size / 2) {
                        0.0
                    } else {
                        frequency[2 * k + 1] as f64 * if bin == k { 1.0 } else { -1.0 }
                    };
                    let angle = std::f64::consts::TAU * (bin * t) as f64 / s.fft_size as f64;
                    value += re * angle.cos() - im * angle.sin();
                }
                signal[frame * s.hop + t] += value / s.fft_size as f64 * w;
                envelope[frame * s.hop + t] += w * w;
            }
        }
        output.extend((pad..pad + s.input_len).map(|t| signal[t] / envelope[t]));
    }
    output
}

#[test]
fn centered_audio_reconstructs_boundaries_and_matches_direct_dft() {
    for cpu in backends() {
        for s in centered_cases() {
            let x = data(s.input_shape());
            let mut spectrum = Tensor::zeros(s.output_shape()).unwrap();
            let mut restored = Tensor::zeros(s.input_shape()).unwrap();
            let mut forward = CenteredStft::new(cpu, s).unwrap();
            let mut inverse = Istft::new(cpu, s).unwrap();
            assert!(inverse.workspace_bytes() > 0);
            forward.run(&x, &mut spectrum).unwrap();
            close(spectrum.data(), &centered_oracle(s, &x), 4e-6, 3e-5);
            let saved = spectrum.data().to_vec();
            inverse.run(&spectrum, &mut restored).unwrap();
            close(
                restored.data(),
                &x.data().iter().map(|&v| v as f64).collect::<Vec<_>>(),
                2e-6,
                2e-5,
            );
            assert_eq!(spectrum.data(), saved);
            no_allocations(|| {
                forward.run(&x, &mut spectrum).unwrap();
                inverse.run(&spectrum, &mut restored).unwrap();
            });
        }
    }
}

#[test]
fn inverse_audio_matches_modified_spectrum_oracle_and_preserves_input() {
    for cpu in backends() {
        for s in centered_cases() {
            let mut spectrum = data(s.output_shape());
            let mut output = Tensor::zeros(s.input_shape()).unwrap();
            let mut inverse = Istft::new(cpu, s).unwrap();
            for _ in 0..2 {
                let saved = spectrum.data().to_vec();
                let expected = inverse_oracle(s, &spectrum);
                inverse.run(&spectrum, &mut output).unwrap();
                close(output.data(), &expected, 2e-6, 3e-5);
                assert_eq!(spectrum.data(), saved);
                for value in spectrum.data_mut() {
                    *value = -*value * 0.6 + 0.13;
                }
            }
        }
    }
}

#[test]
fn centered_audio_rejects_geometry_gaps_and_wrong_axes() {
    let base = StftSpec {
        batch: 1,
        input_len: 25,
        fft_size: 8,
        hop: 2,
    };
    for bad in [
        StftSpec { batch: 0, ..base },
        StftSpec { hop: 0, ..base },
        StftSpec {
            input_len: 4,
            ..base
        },
        StftSpec {
            fft_size: 1,
            ..base
        },
        StftSpec {
            batch: usize::MAX,
            ..base
        },
        StftSpec {
            input_len: i32::MAX as usize,
            ..base
        },
    ] {
        assert!(bad.centered().is_err());
    }
    let gaps = StftSpec { hop: 8, ..base }.centered().unwrap();
    let tail = StftSpec {
        input_len: 13,
        hop: 7,
        ..base
    }
    .centered()
    .unwrap();
    for cpu in backends() {
        assert!(Istft::new(cpu, gaps).is_err());
        assert!(Istft::new(cpu, tail).is_err());
        let s = base.centered().unwrap();
        let mut inverse = Istft::new(cpu, s).unwrap();
        let spectrum = data(s.output_shape());
        let mut wrong = Tensor::zeros([base.input_len, 1]).unwrap();
        assert!(inverse.run(&spectrum, &mut wrong).is_err());
        let wrong = data([1, s.frames(), 2, s.bins()]);
        assert!(
            inverse
                .run(&wrong, &mut Tensor::zeros(s.input_shape()).unwrap())
                .is_err()
        );
    }
}

#[test]
fn centered_audio_reconstructs_long_stereo_and_partial_final_hops() {
    for cpu in backends() {
        for fft_size in [511, 512, 2048] {
            let s = StftSpec {
                batch: 2,
                input_len: 44101,
                fft_size,
                hop: 128,
            }
            .centered()
            .unwrap();
            let x = data(s.input_shape());
            let mut spectrum = Tensor::zeros(s.output_shape()).unwrap();
            let mut y = Tensor::zeros(s.input_shape()).unwrap();
            CenteredStft::new(cpu, s)
                .unwrap()
                .run(&x, &mut spectrum)
                .unwrap();
            Istft::new(cpu, s).unwrap().run(&spectrum, &mut y).unwrap();
            close(
                y.data(),
                &x.data().iter().map(|&v| v as f64).collect::<Vec<_>>(),
                2e-6,
                2e-5,
            );
        }
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_centered_audio_matches_oracles_preserves_spectrum_and_reuses_plans() {
    use fast_audio::cuda;
    let gpu = cuda::Cuda::new(0).expect("CUDA feature tests require an NVIDIA GPU");
    for s in centered_cases() {
        let x = data(s.input_shape());
        let dx = gpu.upload(&x).unwrap();
        let mut spectrum = gpu.zeros(s.output_shape()).unwrap();
        let mut y = gpu.zeros(s.input_shape()).unwrap();
        let mut forward = cuda::CenteredStft::new(&gpu, s).unwrap();
        let mut inverse = cuda::Istft::new(&gpu, s).unwrap();
        assert!(inverse.workspace_bytes() > 0);
        forward.run(&dx, &mut spectrum).unwrap();
        let saved = gpu.download(&spectrum).unwrap();
        close(saved.data(), &centered_oracle(s, &x), 4e-6, 3e-5);
        inverse.run(&spectrum, &mut y).unwrap();
        close(
            gpu.download(&y).unwrap().data(),
            &x.data().iter().map(|&v| v as f64).collect::<Vec<_>>(),
            2e-6,
            2e-5,
        );
        assert_eq!(gpu.download(&spectrum).unwrap().data(), saved.data());
        let modified = data(s.output_shape());
        gpu.copy_from_host(&modified, &mut spectrum).unwrap();
        inverse.run(&spectrum, &mut y).unwrap();
        close(
            gpu.download(&y).unwrap().data(),
            &inverse_oracle(s, &modified),
            2e-6,
            3e-5,
        );
        assert_eq!(gpu.download(&spectrum).unwrap().data(), modified.data());
        let first = gpu.download(&y).unwrap();
        no_allocations(|| {
            inverse.run(&spectrum, &mut y).unwrap();
        });
        assert_eq!(gpu.download(&y).unwrap().data(), first.data());
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_inverse_audio_rejects_gaps_wrong_shapes_and_other_sessions() {
    use fast_audio::cuda;
    let gpu = cuda::Cuda::new(0).unwrap();
    let base = StftSpec {
        batch: 1,
        input_len: 13,
        fft_size: 8,
        hop: 2,
    };
    assert!(cuda::Istft::new(&gpu, StftSpec { hop: 7, ..base }.centered().unwrap()).is_err());
    let s = base.centered().unwrap();
    let mut inverse = cuda::Istft::new(&gpu, s).unwrap();
    let spectrum = gpu.zeros(s.output_shape()).unwrap();
    let mut wrong = gpu.zeros([base.input_len, 1]).unwrap();
    assert!(inverse.run(&spectrum, &mut wrong).is_err());
    let other = cuda::Cuda::new(0).unwrap();
    let spectrum = other.zeros(s.output_shape()).unwrap();
    let mut y = gpu.zeros(s.input_shape()).unwrap();
    assert!(inverse.run(&spectrum, &mut y).is_err());
    let spectrum = gpu.zeros(s.output_shape()).unwrap();
    drop(gpu);
    inverse.run(&spectrum, &mut y).unwrap();
}

#[test]
fn rejects_invalid_shapes_and_geometry() {
    for dims in [vec![usize::MAX, 2], vec![isize::MAX as usize]] {
        assert!(Shape::new(dims).is_err());
    }
    assert!(Tensor::from_vec([2, 3], vec![0.0; 5]).is_err());
    let base = conv_cases()[0];
    for bad in [
        Conv1dSpec { groups: 0, ..base },
        Conv1dSpec { groups: 3, ..base },
        Conv1dSpec { stride: 0, ..base },
        Conv1dSpec {
            dilation: 0,
            ..base
        },
        Conv1dSpec { kernel: 99, ..base },
        Conv1dSpec {
            padding: usize::MAX,
            ..base
        },
        Conv1dSpec {
            batch: i32::MAX as usize + 1,
            ..base
        },
    ] {
        assert!(Conv1d::new(Cpu::Reference, bad, Activation::Identity).is_err());
    }
    for bad in [
        StftSpec {
            batch: 1,
            input_len: 8,
            fft_size: 16,
            hop: 4,
        },
        StftSpec {
            batch: 1,
            input_len: 16,
            fft_size: 16,
            hop: 0,
        },
        StftSpec {
            batch: usize::MAX,
            input_len: 16,
            fft_size: 16,
            hop: 4,
        },
    ] {
        assert!(Stft::new(Cpu::Reference, bad).is_err());
    }
}

#[test]
fn rectangular_matmul_overwrites_and_matches_f64() {
    for cpu in backends() {
        for (m, n, k) in [(1, 1, 1), (3, 5, 7), (11, 2, 19), (4, 33, 8)] {
            let a = data([m, k]);
            let b = data([k, n]);
            let mut c = data([m, n]);
            let expected: Vec<_> = (0..m * n)
                .map(|i| {
                    (0..k)
                        .map(|p| a.data()[(i / n) * k + p] as f64 * b.data()[p * n + i % n] as f64)
                        .sum()
                })
                .collect();
            for _ in 0..2 {
                cpu.matmul(&a, &b, &mut c).unwrap();
                close(c.data(), &expected, 2e-6, 2e-5);
            }
            no_allocations(|| cpu.matmul(&a, &b, &mut c).unwrap());
        }
    }
}

#[test]
fn matmul_rejects_bad_contracts() {
    let a = data([2, 3]);
    let b = data([4, 5]);
    let mut c = data([2, 5]);
    assert!(Cpu::Reference.matmul(&a, &b, &mut c).is_err());
    assert!(
        Cpu::Reference
            .matmul(&data([3]), &data([3, 2]), &mut c)
            .is_err()
    );
    assert!(
        Cpu::Reference
            .matmul(&a, &data([3, 5]), &mut data([10]))
            .is_err()
    );
}

#[test]
fn grouped_strided_dilated_and_pointwise_conv_match_f64_without_allocations() {
    for cpu in backends() {
        for s in conv_cases() {
            for bias_enabled in [false, true] {
                for relu in [false, true] {
                    let x = data(s.input_shape());
                    let mut w = data(s.weight_shape());
                    let bias = bias_enabled.then(|| data([s.out_channels]));
                    let mut y = data(s.output_shape().unwrap());
                    let mut plan = Conv1d::new(
                        cpu,
                        s,
                        if relu {
                            Activation::Relu
                        } else {
                            Activation::Identity
                        },
                    )
                    .unwrap();
                    if s.kernel == 1 && s.padding == 0 {
                        assert_eq!(plan.workspace_bytes(), 0);
                    }
                    for _ in 0..2 {
                        let expected = conv_oracle(s, &x, &w, bias.as_ref(), relu);
                        plan.run(&x, &w, bias.as_ref(), &mut y).unwrap();
                        close(y.data(), &expected, 3e-6, 3e-5);
                        for value in w.data_mut() {
                            *value *= -0.5;
                        }
                    }
                    no_allocations(|| plan.run(&x, &w, bias.as_ref(), &mut y).unwrap());
                    assert!(
                        plan.run(&x, &w, Some(&data([s.out_channels + 1])), &mut y)
                            .is_err()
                    );
                }
            }
        }
    }
}

#[test]
fn odd_even_and_non_power_of_two_stft_match_dft_without_allocations() {
    for cpu in backends() {
        for fft_size in [2, 7, 8, 15, 32] {
            let s = StftSpec {
                batch: 2,
                input_len: 3 * fft_size + 1,
                fft_size,
                hop: fft_size / 2 + 1,
            };
            let x = data(s.input_shape());
            let mut y = Tensor::zeros(s.output_shape().unwrap()).unwrap();
            let mut op = Stft::new(cpu, s).unwrap();
            let expected = stft_oracle(s, &x);
            for _ in 0..2 {
                op.run(&x, &mut y).unwrap();
                close(y.data(), &expected, 3e-6, 1e-5);
            }
            no_allocations(|| op.run(&x, &mut y).unwrap());
            assert!(op.run(&data([1, s.input_len]), &mut y).is_err());
        }
    }
}

#[test]
fn relu_preserves_nan() {
    let s = Conv1dSpec {
        batch: 1,
        in_channels: 1,
        out_channels: 1,
        input_len: 1,
        kernel: 1,
        stride: 1,
        padding: 0,
        dilation: 1,
        groups: 1,
    };
    for cpu in backends() {
        let mut op = Conv1d::new(cpu, s, Activation::Relu).unwrap();
        let mut y = data([1, 1, 1]);
        op.run(
            &Tensor::from_vec([1, 1, 1], vec![f32::NAN]).unwrap(),
            &data([1, 1, 1]),
            None,
            &mut y,
        )
        .unwrap();
        assert!(y.data()[0].is_nan());
    }
}

#[test]
fn prepared_cpu_convolution_snapshots_parameters_and_selects_pointwise_gemm() {
    use fast_audio::cpu::PreparedConv1d;
    for cpu in backends() {
        for s in conv_cases() {
            let x = data(s.input_shape());
            let mut w = data(s.weight_shape());
            let mut bias = data([s.out_channels]);
            let expected = conv_oracle(s, &x, &w, Some(&bias), true);
            let mut reversed = x.clone();
            reversed.data_mut().reverse();
            let later = conv_oracle(s, &reversed, &w, Some(&bias), true);
            let mut plan = PreparedConv1d::new(cpu, s, &w, Some(&bias), Activation::Relu).unwrap();
            w.data_mut().fill(9.0);
            bias.data_mut().fill(-99.0);
            drop(w);
            drop(bias);
            let mut y = data(s.output_shape().unwrap());
            plan.run(&x, &mut y).unwrap();
            close(y.data(), &expected, 5e-6, 3e-5);
            no_allocations(|| plan.run(&reversed, &mut y).unwrap());
            close(y.data(), &later, 5e-6, 3e-5);
            if s.kernel == 1 && s.padding == 0 {
                assert_eq!(plan.workspace_bytes(), 0);
                #[cfg(feature = "mkl")]
                if matches!(cpu, Cpu::Mkl { .. }) {
                    assert!(plan.implementation().starts_with("oneMKL"));
                }
            }
        }
    }
}

#[cfg(feature = "onednn")]
#[test]
fn onednn_packed_parameters_match_oracle_and_remain_independent_of_source_tensors() {
    use fast_audio::onednn;
    for s in conv_cases() {
        for has_bias in [false, true] {
            for relu in [false, true] {
                let mut x = data(s.input_shape());
                let mut w = data(s.weight_shape());
                let mut bias = has_bias.then(|| data([s.out_channels]));
                let expected = conv_oracle(s, &x, &w, bias.as_ref(), relu);
                let mut plan = onednn::Conv1d::new(
                    s,
                    &w,
                    bias.as_ref(),
                    if relu {
                        Activation::Relu
                    } else {
                        Activation::Identity
                    },
                    1,
                )
                .unwrap();
                assert!(!plan.implementation().is_empty());
                assert!(plan.parameter_bytes() >= w.data().len() * 4);
                w.data_mut().fill(99.0);
                if let Some(bias) = &mut bias {
                    bias.data_mut().fill(-999.0);
                }
                drop(w);
                drop(bias);
                let mut y = data(s.output_shape().unwrap());
                plan.run(&x, &mut y).unwrap();
                close(y.data(), &expected, 5e-6, 3e-5);
                no_allocations(|| plan.run(&x, &mut y).unwrap());
                x.data_mut().fill(0.0);
                plan.run(&x, &mut y).unwrap();
                assert!(y.data().iter().all(|v| v.is_finite()));
                assert!(
                    plan.run(&data([s.batch, s.in_channels, s.input_len + 1]), &mut y)
                        .is_err()
                );
                assert!(plan.run(&x, &mut data([y.data().len()])).is_err());
            }
        }
    }
    let s = conv_cases()[0];
    assert!(onednn::Conv1d::new(s, &data([1]), None, Activation::Identity, 1).is_err());
    assert!(
        onednn::Conv1d::new(
            s,
            &data(s.weight_shape()),
            Some(&data([1])),
            Activation::Identity,
            1
        )
        .is_err()
    );
    assert!(
        onednn::Conv1d::new(s, &data(s.weight_shape()), None, Activation::Identity, 0).is_err()
    );
}

#[cfg(feature = "onednn")]
#[test]
fn onednn_restores_calling_threads_openmp_policy() {
    unsafe extern "C" {
        fn omp_get_max_threads() -> i32;
        fn omp_set_num_threads(threads: i32);
    }
    // SAFETY: these OpenMP calls read/set the current host task's ICV, using positive counts.
    let previous = unsafe {
        let n = omp_get_max_threads();
        omp_set_num_threads(3);
        n
    };
    let s = conv_cases()[0];
    let mut plan =
        fast_audio::onednn::Conv1d::new(s, &data(s.weight_shape()), None, Activation::Identity, 1)
            .unwrap();
    // SAFETY: a scalar query for the current task; no pointer arguments or shared writes.
    assert_eq!(unsafe { omp_get_max_threads() }, 3);
    plan.run(&data(s.input_shape()), &mut data(s.output_shape().unwrap()))
        .unwrap();
    // SAFETY: same scalar query as above.
    assert_eq!(unsafe { omp_get_max_threads() }, 3);
    // SAFETY: restore the positive thread count read before this test.
    unsafe { omp_set_num_threads(previous) };
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_primitives_match_independent_oracles_and_reject_foreign_sessions() {
    use fast_audio::cuda::{self, Cuda};
    let gpu = Cuda::new(0).expect("CUDA feature tests require a working GPU");
    let other = Cuda::new(0).unwrap();
    let a = data([3, 7]);
    let b = data([7, 5]);
    let mut expected = data([3, 5]);
    Cpu::Reference.matmul(&a, &b, &mut expected).unwrap();
    let da = gpu.upload(&a).unwrap();
    let db = gpu.upload(&b).unwrap();
    let mut dc = gpu.zeros([3, 5]).unwrap();
    gpu.matmul(&da, &db, &mut dc).unwrap();
    close(
        gpu.download(&dc).unwrap().data(),
        &expected
            .data()
            .iter()
            .map(|&v| v as f64)
            .collect::<Vec<_>>(),
        2e-5,
        2e-5,
    );
    assert!(other.download(&da).is_err());
    assert!(other.matmul(&da, &db, &mut dc).is_err());
    assert!(
        gpu.matmul(&da, &db, &mut gpu.zeros([5, 3]).unwrap())
            .is_err()
    );
    assert!(
        gpu.copy_from_host(&data([1, 21]), &mut gpu.zeros([3, 7]).unwrap())
            .is_err()
    );
    assert_eq!(
        gpu.download(&gpu.zeros([7]).unwrap()).unwrap().data(),
        &[0.0; 7]
    );
    assert!(gpu.measure(0, || Ok(())).is_err());
    for s in conv_cases() {
        for relu in [false, true] {
            for has_bias in [false, true] {
                let x = data(s.input_shape());
                let w = data(s.weight_shape());
                let bias = has_bias.then(|| data([s.out_channels]));
                let expected = conv_oracle(s, &x, &w, bias.as_ref(), relu);
                let x = gpu.upload(&x).unwrap();
                let w = gpu.upload(&w).unwrap();
                let bias = bias.as_ref().map(|b| gpu.upload(b).unwrap());
                let mut y = gpu.zeros(s.output_shape().unwrap()).unwrap();
                let mut op = cuda::Conv1d::new(
                    &gpu,
                    s,
                    if relu {
                        Activation::Relu
                    } else {
                        Activation::Identity
                    },
                )
                .unwrap();
                for _ in 0..2 {
                    op.run(&x, &w, bias.as_ref(), &mut y).unwrap();
                }
                close(gpu.download(&y).unwrap().data(), &expected, 5e-6, 3e-5);
            }
        }
    }
    for fft_size in [2, 7, 8, 15, 32] {
        let s = StftSpec {
            batch: 2,
            input_len: 3 * fft_size + 1,
            fft_size,
            hop: fft_size / 2 + 1,
        };
        let x = data(s.input_shape());
        let expected = stft_oracle(s, &x);
        let x = gpu.upload(&x).unwrap();
        let mut y = gpu.zeros(s.output_shape().unwrap()).unwrap();
        let mut op = cuda::Stft::new(&gpu, s).unwrap();
        for _ in 0..2 {
            op.run(&x, &mut y).unwrap();
        }
        close(gpu.download(&y).unwrap().data(), &expected, 3e-6, 2e-5);
    }
    let clone = gpu.clone();
    drop(gpu);
    assert_eq!(clone.download(&da).unwrap().data(), a.data());
}

#[cfg(feature = "cudnn")]
#[test]
fn cudnn_graphs_match_oracles_and_rebind_without_retaining_borrowed_storage() {
    use fast_audio::cuda::{Conv1d, ConvolutionPolicy, Cuda};
    assert_eq!(
        Cuda::new(0).unwrap().convolution_policy(),
        ConvolutionPolicy::AllowNonDeterministic
    );
    for policy in [
        ConvolutionPolicy::AllowNonDeterministic,
        ConvolutionPolicy::RequireDeterministic,
    ] {
        let gpu = Cuda::with_convolution_policy(0, policy)
            .expect("cuDNN feature tests require the NVIDIA backend");
        assert_eq!(gpu.clone().convolution_policy(), policy);
        for s in conv_cases() {
            for has_bias in [false, true] {
                for relu in [false, true] {
                    let mut x = data(s.input_shape());
                    let mut w = data(s.weight_shape());
                    let bias = has_bias.then(|| data([s.out_channels]));
                    let expected = conv_oracle(s, &x, &w, bias.as_ref(), relu);
                    let dx = gpu.upload(&x).unwrap();
                    let mut dw = gpu.upload(&w).unwrap();
                    let db = bias.as_ref().map(|b| gpu.upload(b).unwrap());
                    let mut y = gpu.zeros(s.output_shape().unwrap()).unwrap();
                    let mut plan = Conv1d::with_cudnn(
                        &gpu,
                        s,
                        has_bias,
                        if relu {
                            Activation::Relu
                        } else {
                            Activation::Identity
                        },
                        64 * 1024 * 1024,
                    )
                    .unwrap();
                    assert!(plan.workspace_bytes() <= 64 * 1024 * 1024);
                    let info = plan.cudnn_plan_info().unwrap();
                    assert_eq!(info.workspace_bytes, plan.workspace_bytes());
                    assert!(info.engine_index >= 0);
                    if policy == ConvolutionPolicy::RequireDeterministic {
                        assert!(info.deterministic);
                    }
                    assert!(plan.implementation().starts_with("cuDNN"));
                    plan.run(&dx, &dw, db.as_ref(), &mut y).unwrap();
                    let first = gpu.download(&y).unwrap();
                    close(first.data(), &expected, 5e-6, 3e-5);
                    no_allocations(|| plan.run(&dx, &dw, db.as_ref(), &mut y).unwrap());
                    if policy == ConvolutionPolicy::RequireDeterministic {
                        assert!(
                            first.data().iter().map(|x| x.to_bits()).eq(gpu
                                .download(&y)
                                .unwrap()
                                .data()
                                .iter()
                                .map(|x| x.to_bits()))
                        );
                    }
                    // Same parameter address, different values: graph plans must not cache weights.
                    for v in w.data_mut() {
                        *v *= -0.5;
                    }
                    gpu.copy_from_host(&w, &mut dw).unwrap();
                    let expected = conv_oracle(s, &x, &w, bias.as_ref(), relu);
                    plan.run(&dx, &dw, db.as_ref(), &mut y).unwrap();
                    close(gpu.download(&y).unwrap().data(), &expected, 5e-6, 3e-5);
                    // New buffer addresses must rebuild an immutable variant pack safely.
                    x.data_mut().reverse();
                    let expected = conv_oracle(s, &x, &w, bias.as_ref(), relu);
                    let new_x = gpu.upload(&x).unwrap();
                    let mut new_y = gpu.zeros(s.output_shape().unwrap()).unwrap();
                    plan.run(&new_x, &dw, db.as_ref(), &mut new_y).unwrap();
                    drop(dx);
                    drop(y);
                    close(gpu.download(&new_y).unwrap().data(), &expected, 5e-6, 3e-5);
                    let mut selected = Conv1d::planned(
                        &gpu,
                        s,
                        has_bias,
                        if relu {
                            Activation::Relu
                        } else {
                            Activation::Identity
                        },
                        64 * 1024 * 1024,
                    )
                    .unwrap();
                    if s.kernel == 1 && s.stride == 1 && s.padding == 0 {
                        assert!(selected.implementation().starts_with("cuBLAS"));
                        assert_eq!(selected.workspace_bytes(), 0);
                        assert!(selected.cudnn_plan_info().is_none());
                    } else {
                        assert!(selected.implementation().starts_with("cuDNN"));
                        if policy == ConvolutionPolicy::RequireDeterministic {
                            assert!(selected.cudnn_plan_info().unwrap().deterministic);
                        }
                    }
                    selected.run(&new_x, &dw, db.as_ref(), &mut new_y).unwrap();
                    close(gpu.download(&new_y).unwrap().data(), &expected, 5e-6, 3e-5);
                    if has_bias {
                        assert!(plan.run(&new_x, &dw, None, &mut new_y).is_err());
                        assert!(selected.run(&new_x, &dw, None, &mut new_y).is_err());
                    } else {
                        let bias = gpu.upload(&data([s.out_channels])).unwrap();
                        assert!(plan.run(&new_x, &dw, Some(&bias), &mut new_y).is_err());
                        assert!(selected.run(&new_x, &dw, Some(&bias), &mut new_y).is_err());
                    }
                }
            }
        }
    }
}

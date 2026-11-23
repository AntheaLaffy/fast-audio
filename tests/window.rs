use fast_audio::{
    StftSpec, Tensor,
    cpu::{CenteredStft, Istft, Stft},
};
mod support;
use support::{backends, close, data, no_allocations};

fn window(n: usize) -> Tensor {
    Tensor::from_vec(
        [n],
        (0..n)
            .map(|i| {
                if i % 2 == 0 {
                    0.25 + i as f32 * 0.08
                } else {
                    -0.8 + i as f32 * 0.03
                }
            })
            .collect(),
    )
    .unwrap()
}
fn forward(s: StftSpec, x: &Tensor, w: &Tensor, centered: bool) -> Vec<f64> {
    let padding = if centered { s.fft_size / 2 } else { 0 };
    let frames = if centered {
        s.centered().unwrap().frames()
    } else {
        s.frames().unwrap()
    };
    let mut out = Vec::new();
    for row in x.data().chunks_exact(s.input_len) {
        let mut signal: Vec<f32> = Vec::new();
        if centered {
            signal.extend(row[1..=padding].iter().rev());
        }
        signal.extend(row);
        if centered {
            signal.extend(row[s.input_len - 1 - padding..s.input_len - 1].iter().rev());
        }
        for frame in 0..frames {
            for bin in 0..s.bins() {
                let (mut re, mut im) = (0.0, 0.0);
                for t in 0..s.fft_size {
                    let value = (signal[frame * s.hop + t] * w.data()[t]) as f64;
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
fn inverse_dc(s: StftSpec, w: &Tensor) -> Vec<f64> {
    let frames = s.centered().unwrap().frames();
    let pad = s.fft_size / 2;
    let row: Vec<_> = (0..s.input_len)
        .map(|t| {
            let (mut numerator, mut denominator) = (0.0, 0.0);
            for frame in 0..frames {
                let q = t as isize + pad as isize - (frame * s.hop) as isize;
                if q >= 0 && q < s.fft_size as isize {
                    let v = w.data()[q as usize] as f64;
                    numerator += v;
                    denominator += v * v;
                }
            }
            numerator / (s.fft_size as f64 * denominator)
        })
        .collect();
    row.repeat(s.batch)
}

#[test]
fn arbitrary_windows_match_dft_dc_inverse_and_snapshot_coefficients() {
    for cpu in backends() {
        for n in [7, 8] {
            let s = StftSpec {
                batch: 2,
                input_len: 29,
                fft_size: n,
                hop: 3,
            };
            let geometry = s.centered().unwrap();
            let mut w = window(n);
            let x = data(s.input_shape());
            let expected = forward(s, &x, &w, true);
            let expected_dc = inverse_dc(s, &w);
            let mut forward_plan = CenteredStft::with_window(cpu, geometry, &w).unwrap();
            let mut inverse = Istft::with_window(cpu, geometry, &w).unwrap();
            let mut raw = Stft::with_window(cpu, s, &w).unwrap();
            let expected_raw = forward(s, &x, &w, false);
            w.data_mut().fill(0.0);
            let mut spectrum = Tensor::zeros(geometry.output_shape()).unwrap();
            let mut output = Tensor::zeros(s.input_shape()).unwrap();
            forward_plan.run(&x, &mut spectrum).unwrap();
            close(spectrum.data(), &expected, 3e-6, 3e-5);
            inverse.run(&spectrum, &mut output).unwrap();
            close(
                output.data(),
                &x.data().iter().map(|&v| v as f64).collect::<Vec<_>>(),
                3e-6,
                3e-5,
            );
            no_allocations(|| {
                forward_plan.run(&x, &mut spectrum).unwrap();
                inverse.run(&spectrum, &mut output).unwrap();
            });
            let mut raw_output = Tensor::zeros(s.output_shape().unwrap()).unwrap();
            raw.run(&x, &mut raw_output).unwrap();
            close(raw_output.data(), &expected_raw, 3e-6, 3e-5);
            spectrum.data_mut().fill(0.0);
            for frame in spectrum.data_mut().chunks_exact_mut(s.bins() * 2) {
                frame[0] = 1.0;
            }
            inverse.run(&spectrum, &mut output).unwrap();
            close(output.data(), &expected_dc, 2e-6, 2e-5);
            assert!(Istft::with_window(cpu, geometry, &w).is_err());
            for bad in [
                Tensor::zeros([1, n]).unwrap(),
                Tensor::from_vec([n], vec![f32::INFINITY; n]).unwrap(),
            ] {
                assert!(CenteredStft::with_window(cpu, geometry, &bad).is_err());
            }
        }
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_custom_window_matches_dft_and_dc_inverse_without_host_fallback() {
    use fast_audio::cuda;
    let gpu = cuda::Cuda::new(0).unwrap();
    for n in [7, 8] {
        let s = StftSpec {
            batch: 2,
            input_len: 29,
            fft_size: n,
            hop: 3,
        };
        let geometry = s.centered().unwrap();
        let w = window(n);
        let x = data(s.input_shape());
        let input = gpu.upload(&x).unwrap();
        let mut plan = cuda::CenteredStft::with_window(&gpu, geometry, &w).unwrap();
        let mut inverse = cuda::Istft::with_window(&gpu, geometry, &w).unwrap();
        let mut spectrum = gpu.zeros(geometry.output_shape()).unwrap();
        let mut output = gpu.zeros(s.input_shape()).unwrap();
        plan.run(&input, &mut spectrum).unwrap();
        close(
            gpu.download(&spectrum).unwrap().data(),
            &forward(s, &x, &w, true),
            3e-6,
            3e-5,
        );
        inverse.run(&spectrum, &mut output).unwrap();
        close(
            gpu.download(&output).unwrap().data(),
            &x.data().iter().map(|&v| v as f64).collect::<Vec<_>>(),
            3e-6,
            3e-5,
        );
        let mut dc = Tensor::zeros(geometry.output_shape()).unwrap();
        for frame in dc.data_mut().chunks_exact_mut(s.bins() * 2) {
            frame[0] = 1.0;
        }
        gpu.copy_from_host(&dc, &mut spectrum).unwrap();
        inverse.run(&spectrum, &mut output).unwrap();
        close(
            gpu.download(&output).unwrap().data(),
            &inverse_dc(s, &w),
            2e-6,
            2e-5,
        );
        no_allocations(|| inverse.run(&spectrum, &mut output).unwrap());
    }
}

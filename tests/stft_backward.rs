#[allow(dead_code)]
mod support;

use fast_audio::{
    BinaryOp, GradientMode, Layout, ReduceOp, Shared, StftSpec, Tensor, UnaryOp,
    autograd::{Binding, Builder},
    cpu::{
        CenteredStftBackward, CenteredStftPlan, Cpu, IstftBackward, IstftPlan, StftBackward,
        StftPlan, Tape,
    },
};
use std::f64::consts::TAU;
use support::{backends, close, no_allocations};

fn signal(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed ^ 0x9e37_79b9_7f4a_7c15;
    (0..n)
        .map(|i| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((s >> 33) as f64) / ((1u64 << 31) as f64) - 0.5;
            (u * 1.4 + 0.25 * (i as f64 * 0.37).sin()) as f32
        })
        .collect()
}

fn hann_window(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| 0.5 - 0.5 * (TAU * i as f64 / n as f64).cos())
        .collect()
}

fn custom_window(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let phase = i as f64 / n as f64;
            (0.65 - 0.35 * (TAU * phase).cos() + 0.18 * (TAU * phase).sin() - 0.05 * phase) as f32
        })
        .collect()
}

fn close_f32(actual: &[f32], expected: &[f32], atol: f64, rtol: f64) {
    let exp: Vec<f64> = expected.iter().map(|&v| v as f64).collect();
    close(actual, &exp, atol, rtol);
}

/// Independent FP64 DFT forward oracle for uncentered (`padding = 0`) and centered (`padding = n/2`).
#[allow(clippy::too_many_arguments)]
fn ref_stft_f64(
    x: &[f64],
    batch: usize,
    input_len: usize,
    n: usize,
    hop: usize,
    padding: usize,
    frames: usize,
    window: &[f64],
) -> Vec<f64> {
    let bins = n / 2 + 1;
    let mut out = vec![0.0f64; batch * frames * bins * 2];
    for b in 0..batch {
        for f in 0..frames {
            let start = f * hop;
            let mut framed = vec![0.0f64; n];
            for t in 0..n {
                let pos = start + t;
                let src = if padding == 0 {
                    pos
                } else if pos < padding {
                    padding - pos
                } else {
                    let q = pos - padding;
                    if q < input_len {
                        q
                    } else {
                        2 * (input_len - 1) - q
                    }
                };
                framed[t] = x[b * input_len + src] * window[t];
            }
            for k in 0..bins {
                let mut re = 0.0f64;
                let mut im = 0.0f64;
                for (t, &u) in framed.iter().enumerate() {
                    let theta = TAU * (k * t) as f64 / n as f64;
                    re += u * theta.cos();
                    im -= u * theta.sin();
                }
                let idx = ((b * frames + f) * bins + k) * 2;
                out[idx] = re;
                out[idx + 1] = im;
            }
        }
    }
    out
}

/// Independent FP64 exact transpose oracle for STFT/CenteredSTFT.
#[allow(clippy::too_many_arguments)]
fn ref_stft_backward_f64(
    dy: &[f64],
    batch: usize,
    input_len: usize,
    n: usize,
    hop: usize,
    padding: usize,
    frames: usize,
    window: &[f64],
) -> Vec<f64> {
    let bins = n / 2 + 1;
    let mut dx = vec![0.0f64; batch * input_len];
    for b in 0..batch {
        for f in 0..frames {
            let start = f * hop;
            for (t, &w) in window.iter().enumerate() {
                let mut du = 0.0f64;
                for k in 0..bins {
                    let idx = ((b * frames + f) * bins + k) * 2;
                    let theta = TAU * (k * t) as f64 / n as f64;
                    du += dy[idx] * theta.cos() - dy[idx + 1] * theta.sin();
                }
                let pos = start + t;
                let src = if padding == 0 {
                    pos
                } else if pos < padding {
                    padding - pos
                } else {
                    let q = pos - padding;
                    if q < input_len {
                        q
                    } else {
                        2 * (input_len - 1) - q
                    }
                };
                dx[b * input_len + src] += du * w;
            }
        }
    }
    dx
}

/// Independent FP64 centered iSTFT forward oracle.
#[allow(clippy::too_many_arguments)]
fn ref_istft_f64(
    s: &[f64],
    batch: usize,
    output_len: usize,
    n: usize,
    hop: usize,
    padding: usize,
    frames: usize,
    window: &[f64],
) -> Vec<f64> {
    let bins = n / 2 + 1;
    let even_n = n.is_multiple_of(2);
    let mut envelope = vec![0.0f64; output_len];
    for f in 0..frames {
        for (t, &w) in window.iter().enumerate() {
            let pos = f * hop + t;
            if pos >= padding && pos - padding < output_len {
                envelope[pos - padding] += w * w;
            }
        }
    }
    let mut out = vec![0.0f64; batch * output_len];
    for b in 0..batch {
        for f in 0..frames {
            let start = f * hop;
            for (t, &w) in window.iter().enumerate() {
                let pos = start + t;
                if pos < padding || pos - padding >= output_len {
                    continue;
                }
                let mut v = 0.0f64;
                for k in 0..bins {
                    let idx = ((b * frames + f) * bins + k) * 2;
                    let theta = TAU * (k * t) as f64 / n as f64;
                    let re = s[idx];
                    let im = if k == 0 || (even_n && k == n / 2) {
                        0.0
                    } else {
                        s[idx + 1]
                    };
                    if k == 0 || (even_n && k == n / 2) {
                        v += re * theta.cos();
                    } else {
                        v += 2.0 * (re * theta.cos() - im * theta.sin());
                    }
                }
                out[b * output_len + (pos - padding)] += v * w;
            }
        }
        for sample in 0..output_len {
            out[b * output_len + sample] /= n as f64 * envelope[sample];
        }
    }
    out
}

/// Independent FP64 exact transpose oracle for centered iSTFT.
#[allow(clippy::too_many_arguments)]
fn ref_istft_backward_f64(
    dy: &[f64],
    batch: usize,
    output_len: usize,
    n: usize,
    hop: usize,
    padding: usize,
    frames: usize,
    window: &[f64],
) -> Vec<f64> {
    let bins = n / 2 + 1;
    let even_n = n.is_multiple_of(2);
    let mut envelope = vec![0.0f64; output_len];
    for f in 0..frames {
        for (t, &w) in window.iter().enumerate() {
            let pos = f * hop + t;
            if pos >= padding && pos - padding < output_len {
                envelope[pos - padding] += w * w;
            }
        }
    }
    let mut ds = vec![0.0f64; batch * frames * bins * 2];
    for b in 0..batch {
        for f in 0..frames {
            let start = f * hop;
            for k in 0..bins {
                let endpoint = k == 0 || (even_n && k == n / 2);
                let mut re = 0.0f64;
                let mut im = 0.0f64;
                for (t, &w) in window.iter().enumerate() {
                    let pos = start + t;
                    if pos < padding || pos - padding >= output_len {
                        continue;
                    }
                    let sample = pos - padding;
                    let dv = dy[b * output_len + sample] * w / (n as f64 * envelope[sample]);
                    let theta = TAU * (k * t) as f64 / n as f64;
                    if endpoint {
                        re += dv * theta.cos();
                    } else {
                        re += 2.0 * dv * theta.cos();
                        im -= 2.0 * dv * theta.sin();
                    }
                }
                let idx = ((b * frames + f) * bins + k) * 2;
                ds[idx] = re;
                ds[idx + 1] = im;
            }
        }
    }
    ds
}

fn dot_f64(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(&x, &y)| x as f64 * y as f64).sum()
}

#[test]
fn stft_and_centered_stft_forward_backward_match_fp64_and_adjoint() -> fast_audio::Result<()> {
    for cpu in backends() {
        // Uncentered cases: even, odd, non-hop-aligned tail, and hop > fft_size gap case.
        let uncentered_cases = [
            (2, 20, 8, 4, false),
            (2, 19, 7, 3, true),
            (1, 20, 8, 10, false),
            (3, 17, 6, 2, true),
        ];
        for (ci, &(batch, input_len, fft_size, hop, use_custom)) in
            uncentered_cases.iter().enumerate()
        {
            let spec = StftSpec {
                batch,
                input_len,
                fft_size,
                hop,
            };
            let frames = spec.frames()?;
            let bins = spec.bins();
            let x_vec = signal(batch * input_len, 100 + ci as u64);
            let dy_vec = signal(batch * frames * bins * 2, 200 + ci as u64);
            let win_f32 = if use_custom {
                Some(Tensor::from_vec([fft_size], custom_window(fft_size))?)
            } else {
                None
            };
            let win_f64: Vec<f64> = match &win_f32 {
                Some(t) => t.data().iter().map(|&v| v as f64).collect(),
                None => hann_window(fft_size),
            };
            let x_f64: Vec<f64> = x_vec.iter().map(|&v| v as f64).collect();
            let dy_f64: Vec<f64> = dy_vec.iter().map(|&v| v as f64).collect();
            let ref_y = ref_stft_f64(&x_f64, batch, input_len, fft_size, hop, 0, frames, &win_f64);
            let ref_dx = ref_stft_backward_f64(
                &dy_f64, batch, input_len, fft_size, hop, 0, frames, &win_f64,
            );

            let x_t = Tensor::from_vec([batch, input_len], x_vec.clone())?;
            let dy_t = Tensor::from_vec([batch, frames, bins, 2], dy_vec.clone())?;
            let mut y_t = Tensor::zeros([batch, frames, bins, 2])?;
            let mut dx_t = Tensor::zeros([batch, input_len])?;
            let x_lay = Layout::contiguous([batch, input_len])?;
            let y_lay = Layout::contiguous([batch, frames, bins, 2])?;

            let mut fwd = match &win_f32 {
                Some(w) => StftPlan::with_window(cpu, spec, w, &x_lay, &y_lay)?,
                None => StftPlan::new(cpu, spec, &x_lay, &y_lay)?,
            };
            let mut bwd = match &win_f32 {
                Some(w) => StftBackward::with_window(cpu, spec, w, &y_lay, &x_lay)?,
                None => StftBackward::new(cpu, spec, &y_lay, &x_lay)?,
            };
            assert!(!bwd.input_required());
            assert!(fwd.workspace_bytes() > 0 && bwd.workspace_bytes() > 0);

            // Warm once, then verify zero hot Rust heap allocations.
            fwd.run(&x_t.view(&x_lay)?, &mut y_t.view_mut(&y_lay)?)?;
            bwd.run(
                &dy_t.view(&y_lay)?,
                &mut dx_t.view_mut(&x_lay)?,
                GradientMode::Overwrite,
            )?;
            no_allocations(|| {
                fwd.run(
                    &x_t.view(&x_lay).unwrap(),
                    &mut y_t.view_mut(&y_lay).unwrap(),
                )
                .unwrap();
                bwd.run(
                    &dy_t.view(&y_lay).unwrap(),
                    &mut dx_t.view_mut(&x_lay).unwrap(),
                    GradientMode::Overwrite,
                )
                .unwrap();
            });
            close(y_t.data(), &ref_y, 2e-5, 2e-5);
            close(dx_t.data(), &ref_dx, 2e-5, 2e-5);

            // Inner-product adjoint identity <dY, F(x)> == <F*(dY), x>.
            let lhs = dot_f64(&dy_vec, y_t.data());
            let rhs = dot_f64(dx_t.data(), &x_vec);
            assert!(
                (lhs - rhs).abs() <= 2e-5 * (1.0 + lhs.abs().max(rhs.abs())),
                "adjoint mismatch: lhs={lhs} rhs={rhs}"
            );

            // Verify Accumulate mode adds onto existing buffer.
            let init_grad = signal(batch * input_len, 300 + ci as u64);
            let mut acc_t = Tensor::from_vec([batch, input_len], init_grad.clone())?;
            bwd.run(
                &dy_t.view(&y_lay)?,
                &mut acc_t.view_mut(&x_lay)?,
                GradientMode::Accumulate,
            )?;
            let expected_acc: Vec<f64> = init_grad
                .iter()
                .zip(&ref_dx)
                .map(|(&a, &b)| a as f64 + b)
                .collect();
            close(acc_t.data(), &expected_acc, 2e-5, 2e-5);
        }

        // Centered cases including minimal reflection overlap L = P + 1 (L=5, N=8, P=4) and odd N=7.
        let centered_cases = [
            (2, 20, 8, 4, false),
            (2, 15, 7, 3, true),
            (2, 5, 8, 2, true),
            (2, 4, 6, 2, false),
        ];
        for (ci, &(batch, input_len, fft_size, hop, use_custom)) in
            centered_cases.iter().enumerate()
        {
            let spec = StftSpec {
                batch,
                input_len,
                fft_size,
                hop,
            }
            .centered()?;
            let frames = spec.frames();
            let bins = spec.bins();
            let padding = fft_size / 2;
            let x_vec = signal(batch * input_len, 400 + ci as u64);
            let dy_vec = signal(batch * frames * bins * 2, 500 + ci as u64);
            let mut win_f32 = if use_custom {
                Some(Tensor::from_vec([fft_size], custom_window(fft_size))?)
            } else {
                None
            };
            let win_f64: Vec<f64> = match &win_f32 {
                Some(t) => t.data().iter().map(|&v| v as f64).collect(),
                None => hann_window(fft_size),
            };
            let x_f64: Vec<f64> = x_vec.iter().map(|&v| v as f64).collect();
            let dy_f64: Vec<f64> = dy_vec.iter().map(|&v| v as f64).collect();
            let ref_y = ref_stft_f64(
                &x_f64, batch, input_len, fft_size, hop, padding, frames, &win_f64,
            );
            let ref_dx = ref_stft_backward_f64(
                &dy_f64, batch, input_len, fft_size, hop, padding, frames, &win_f64,
            );

            // Exercise strided & reversed views: reversed sample axis + permuted spectrum [B, K, F, 2].
            let x_base_lay = Layout::contiguous([batch, input_len])?;
            let x_rev_lay = x_base_lay.slice(1, input_len - 1, input_len, -1)?;
            let y_perm_storage = Layout::contiguous([batch, bins, frames, 2])?;
            let y_view_lay = y_perm_storage.permute(&[0, 2, 1, 3])?;

            let mut x_storage = Tensor::zeros([batch, input_len])?;
            for b in 0..batch {
                for s in 0..input_len {
                    *x_storage.view_mut(&x_rev_lay)?.get_mut(&[b, s]).unwrap() =
                        x_vec[b * input_len + s];
                }
            }
            let mut dy_storage = Tensor::zeros([batch, bins, frames, 2])?;
            for b in 0..batch {
                for f in 0..frames {
                    for k in 0..bins {
                        let base = ((b * frames + f) * bins + k) * 2;
                        *dy_storage
                            .view_mut(&y_view_lay)?
                            .get_mut(&[b, f, k, 0])
                            .unwrap() = dy_vec[base];
                        *dy_storage
                            .view_mut(&y_view_lay)?
                            .get_mut(&[b, f, k, 1])
                            .unwrap() = dy_vec[base + 1];
                    }
                }
            }
            let mut y_storage = Tensor::zeros([batch, bins, frames, 2])?;
            let mut dx_storage = Tensor::zeros([batch, input_len])?;

            let mut fwd = match &win_f32 {
                Some(w) => CenteredStftPlan::with_window(cpu, spec, w, &x_rev_lay, &y_view_lay)?,
                None => CenteredStftPlan::new(cpu, spec, &x_rev_lay, &y_view_lay)?,
            };
            let mut bwd = match &win_f32 {
                Some(w) => {
                    CenteredStftBackward::with_window(cpu, spec, w, &y_view_lay, &x_rev_lay)?
                }
                None => CenteredStftBackward::new(cpu, spec, &y_view_lay, &x_rev_lay)?,
            };
            // Mutating caller window tensor after plan construction must NOT affect plan output.
            if let Some(w) = win_f32.as_mut() {
                w.data_mut().fill(999.0);
            }

            fwd.run(
                &x_storage.view(&x_rev_lay)?,
                &mut y_storage.view_mut(&y_view_lay)?,
            )?;
            bwd.run(
                &dy_storage.view(&y_view_lay)?,
                &mut dx_storage.view_mut(&x_rev_lay)?,
                GradientMode::Overwrite,
            )?;
            let y_mat = y_storage.view(&y_view_lay)?.to_tensor()?;
            let dx_mat = dx_storage.view(&x_rev_lay)?.to_tensor()?;
            close(y_mat.data(), &ref_y, 2e-5, 2e-5);
            close(dx_mat.data(), &ref_dx, 2e-5, 2e-5);

            let lhs = dot_f64(&dy_vec, y_mat.data());
            let rhs = dot_f64(dx_mat.data(), &x_vec);
            assert!(
                (lhs - rhs).abs() <= 2e-5 * (1.0 + lhs.abs().max(rhs.abs())),
                "centered adjoint mismatch: lhs={lhs} rhs={rhs}"
            );
        }
    }
    Ok(())
}

#[test]
fn istft_forward_backward_match_fp64_adjoint_and_ignore_endpoint_imag() -> fast_audio::Result<()> {
    for cpu in backends() {
        // Reject NOLA violation (periodic Hann has w[0]=0, so hop==fft_size leaves uncovered samples).
        let bad_spec = StftSpec {
            batch: 1,
            input_len: 16,
            fft_size: 8,
            hop: 8,
        }
        .centered()?;
        assert!(
            IstftPlan::new(
                cpu,
                bad_spec,
                &Layout::contiguous(bad_spec.output_shape())?,
                &Layout::contiguous(bad_spec.input_shape())?,
            )
            .is_err()
        );
        assert!(
            IstftBackward::new(
                cpu,
                bad_spec,
                &Layout::contiguous(bad_spec.input_shape())?,
                &Layout::contiguous(bad_spec.output_shape())?,
            )
            .is_err()
        );

        let cases = [
            (2, 20, 8, 4, false),
            (2, 17, 7, 3, true),
            (2, 9, 8, 2, true),
        ];
        for (ci, &(batch, output_len, fft_size, hop, use_custom)) in cases.iter().enumerate() {
            let spec = StftSpec {
                batch,
                input_len: output_len,
                fft_size,
                hop,
            }
            .centered()?;
            let frames = spec.frames();
            let bins = spec.bins();
            let padding = fft_size / 2;
            let mut s_vec = signal(batch * frames * bins * 2, 600 + ci as u64);
            // Inject large non-zero imaginary parts at DC and even-N Nyquist to verify endpoint policy.
            for b in 0..batch {
                for f in 0..frames {
                    let dc = ((b * frames + f) * bins) * 2;
                    s_vec[dc + 1] = 17.25;
                    if fft_size % 2 == 0 {
                        let nyq = ((b * frames + f) * bins + (bins - 1)) * 2;
                        s_vec[nyq + 1] = -23.5;
                    }
                }
            }
            let dy_vec = signal(batch * output_len, 700 + ci as u64);
            let win_f32 = if use_custom {
                Some(Tensor::from_vec([fft_size], custom_window(fft_size))?)
            } else {
                None
            };
            let win_f64: Vec<f64> = match &win_f32 {
                Some(t) => t.data().iter().map(|&v| v as f64).collect(),
                None => hann_window(fft_size),
            };
            let s_f64: Vec<f64> = s_vec.iter().map(|&v| v as f64).collect();
            let dy_f64: Vec<f64> = dy_vec.iter().map(|&v| v as f64).collect();
            let ref_y = ref_istft_f64(
                &s_f64, batch, output_len, fft_size, hop, padding, frames, &win_f64,
            );
            let ref_ds = ref_istft_backward_f64(
                &dy_f64, batch, output_len, fft_size, hop, padding, frames, &win_f64,
            );

            // Test contiguous path with zero allocations and non-mutated input S.
            let s_lay = Layout::contiguous([batch, frames, bins, 2])?;
            let y_lay = Layout::contiguous([batch, output_len])?;
            let s_t = Tensor::from_vec([batch, frames, bins, 2], s_vec.clone())?;
            let dy_t = Tensor::from_vec([batch, output_len], dy_vec.clone())?;
            let mut y_t = Tensor::zeros([batch, output_len])?;
            let mut ds_t = Tensor::zeros([batch, frames, bins, 2])?;

            let mut fwd = match &win_f32 {
                Some(w) => IstftPlan::with_window(cpu, spec, w, &s_lay, &y_lay)?,
                None => IstftPlan::new(cpu, spec, &s_lay, &y_lay)?,
            };
            let mut bwd = match &win_f32 {
                Some(w) => IstftBackward::with_window(cpu, spec, w, &y_lay, &s_lay)?,
                None => IstftBackward::new(cpu, spec, &y_lay, &s_lay)?,
            };
            assert!(!bwd.input_required());

            fwd.run(&s_t.view(&s_lay)?, &mut y_t.view_mut(&y_lay)?)?;
            bwd.run(
                &dy_t.view(&y_lay)?,
                &mut ds_t.view_mut(&s_lay)?,
                GradientMode::Overwrite,
            )?;
            no_allocations(|| {
                fwd.run(
                    &s_t.view(&s_lay).unwrap(),
                    &mut y_t.view_mut(&y_lay).unwrap(),
                )
                .unwrap();
                bwd.run(
                    &dy_t.view(&y_lay).unwrap(),
                    &mut ds_t.view_mut(&s_lay).unwrap(),
                    GradientMode::Overwrite,
                )
                .unwrap();
            });
            // Caller spectrum must remain untouched.
            assert_eq!(s_t.data(), &s_vec);
            close(y_t.data(), &ref_y, 2e-5, 2e-5);
            close(ds_t.data(), &ref_ds, 2e-5, 2e-5);

            // DC and even-N Nyquist imaginary gradients must be exact zero.
            for b in 0..batch {
                for f in 0..frames {
                    let dc = ((b * frames + f) * bins) * 2;
                    assert_eq!(ds_t.data()[dc + 1], 0.0);
                    if fft_size % 2 == 0 {
                        let nyq = ((b * frames + f) * bins + (bins - 1)) * 2;
                        assert_eq!(ds_t.data()[nyq + 1], 0.0);
                    }
                }
            }

            // Adjoint identity <dy, iSTFT(S)> == <iSTFT*(dy), S>.
            let lhs = dot_f64(&dy_vec, y_t.data());
            let rhs = dot_f64(ds_t.data(), &s_vec);
            assert!(
                (lhs - rhs).abs() <= 2e-5 * (1.0 + lhs.abs().max(rhs.abs())),
                "iSTFT adjoint mismatch: lhs={lhs} rhs={rhs}"
            );

            // Strided permuted + reversed views with Accumulate mode.
            let s_perm_storage = Layout::contiguous([batch, bins, frames, 2])?;
            let s_view_lay = s_perm_storage.permute(&[0, 2, 1, 3])?;
            let y_rev_lay = Layout::contiguous([batch, output_len])?.slice(
                1,
                output_len - 1,
                output_len,
                -1,
            )?;
            let mut s_strided = Tensor::zeros([batch, bins, frames, 2])?;
            for b in 0..batch {
                for f in 0..frames {
                    for k in 0..bins {
                        let base = ((b * frames + f) * bins + k) * 2;
                        *s_strided
                            .view_mut(&s_view_lay)?
                            .get_mut(&[b, f, k, 0])
                            .unwrap() = s_vec[base];
                        *s_strided
                            .view_mut(&s_view_lay)?
                            .get_mut(&[b, f, k, 1])
                            .unwrap() = s_vec[base + 1];
                    }
                }
            }
            let mut dy_strided = Tensor::zeros([batch, output_len])?;
            for b in 0..batch {
                for sample in 0..output_len {
                    *dy_strided
                        .view_mut(&y_rev_lay)?
                        .get_mut(&[b, sample])
                        .unwrap() = dy_vec[b * output_len + sample];
                }
            }
            let mut y_strided = Tensor::zeros([batch, output_len])?;
            let mut ds_strided = Tensor::from_vec(
                [batch, bins, frames, 2],
                vec![0.25f32; batch * bins * frames * 2],
            )?;
            let mut fwd_s = match &win_f32 {
                Some(w) => IstftPlan::with_window(cpu, spec, w, &s_view_lay, &y_rev_lay)?,
                None => IstftPlan::new(cpu, spec, &s_view_lay, &y_rev_lay)?,
            };
            let mut bwd_s = match &win_f32 {
                Some(w) => IstftBackward::with_window(cpu, spec, w, &y_rev_lay, &s_view_lay)?,
                None => IstftBackward::new(cpu, spec, &y_rev_lay, &s_view_lay)?,
            };
            fwd_s.run(
                &s_strided.view(&s_view_lay)?,
                &mut y_strided.view_mut(&y_rev_lay)?,
            )?;
            bwd_s.run(
                &dy_strided.view(&y_rev_lay)?,
                &mut ds_strided.view_mut(&s_view_lay)?,
                GradientMode::Accumulate,
            )?;
            close(
                y_strided.view(&y_rev_lay)?.to_tensor()?.data(),
                &ref_y,
                2e-5,
                2e-5,
            );
            let expected_acc: Vec<f64> = ref_ds.iter().map(|v| v + 0.25).collect();
            close(
                ds_strided.view(&s_view_lay)?.to_tensor()?.data(),
                &expected_acc,
                2e-5,
                2e-5,
            );
        }
    }
    Ok(())
}

#[test]
fn stft_centered_istft_tape_training_matches_fp64_and_fuses_tail() -> fast_audio::Result<()> {
    for cpu in backends() {
        let (batch, length, fft_size, hop) = (2, 16, 8, 4);
        let spec = StftSpec {
            batch,
            input_len: length,
            fft_size,
            hop,
        }
        .centered()?;
        let frames = spec.frames();
        let bins = spec.bins();
        let padding = fft_size / 2;
        let win = Tensor::from_vec([fft_size], custom_window(fft_size))?;
        let win_f64: Vec<f64> = win.data().iter().map(|&v| v as f64).collect();

        // Graph: x -> centered_stft -> mask * spec + bias -> istft -> (y - target)^2 -> mean
        let mut builder = Builder::new()?;
        let x = builder.leaf([batch, length], true)?;
        let mask = builder.leaf([1, 1, bins, 2], true)?;
        let bias = builder.leaf([1, 1, bins, 2], true)?;
        let target = builder.leaf([batch, length], false)?;

        let s0 = builder.centered_stft(x, fft_size, hop, Some(&win))?;
        let s1 = builder.binary(BinaryOp::Mul, s0, mask)?;
        let s2 = builder.binary(BinaryOp::Add, s1, bias)?;
        let y = builder.istft(s2, length, fft_size, hop, Some(&win))?;
        let diff = builder.binary(BinaryOp::Sub, y, target)?;
        let sq = builder.unary(UnaryOp::Square, diff)?;
        let loss = builder.reduce(ReduceOp::Mean, sq, &[0, 1], false)?;
        let program = builder.finish(&[loss])?;

        let x_init = signal(batch * length, 801);
        let mask_init = signal(bins * 2, 802);
        let bias_init = signal(bins * 2, 803);
        let target_init = signal(batch * length, 804);

        // 1. Unfused Tape
        let x_u = Shared::new(Tensor::from_vec([batch, length], x_init.clone())?)?;
        let m_u = Shared::new(Tensor::from_vec([1, 1, bins, 2], mask_init.clone())?)?;
        let b_u = Shared::new(Tensor::from_vec([1, 1, bins, 2], bias_init.clone())?)?;
        let t_u = Shared::new(Tensor::from_vec([batch, length], target_init.clone())?)?;
        let mut tape_u = Tape::new(
            cpu,
            &program,
            &[
                Binding::new(x, x_u.clone()),
                Binding::new(mask, m_u.clone()),
                Binding::new(bias, b_u.clone()),
                Binding::new(target, t_u.clone()),
            ],
        )?;

        // 2. Fused Tape (step_scalar fuses post-Istft Sub+Square+Mean tail!)
        let x_f = Shared::new(Tensor::from_vec([batch, length], x_init.clone())?)?;
        let m_f = Shared::new(Tensor::from_vec([1, 1, bins, 2], mask_init.clone())?)?;
        let b_f = Shared::new(Tensor::from_vec([1, 1, bins, 2], bias_init.clone())?)?;
        let t_f = Shared::new(Tensor::from_vec([batch, length], target_init.clone())?)?;
        let mut tape_f = Tape::new(
            cpu,
            &program,
            &[
                Binding::new(x, x_f.clone()),
                Binding::new(mask, m_f.clone()),
                Binding::new(bias, b_f.clone()),
                Binding::new(target, t_f.clone()),
            ],
        )?;

        // 3. Independent FP64 training oracle
        let mut x_ref: Vec<f64> = x_init.iter().map(|&v| v as f64).collect();
        let mut m_ref: Vec<f64> = mask_init.iter().map(|&v| v as f64).collect();
        let mut b_ref: Vec<f64> = bias_init.iter().map(|&v| v as f64).collect();
        let target_ref: Vec<f64> = target_init.iter().map(|&v| v as f64).collect();

        let rate = 0.05f32;
        let params = [x, mask, bias];
        for step in 0..6 {
            tape_u.forward()?;
            tape_u.backward_scalar(GradientMode::Overwrite)?;
            tape_u.sgd(&params, rate)?;

            if step == 0 {
                tape_f.step_scalar(&params, rate)?;
                assert!(
                    tape_f.has_fused_plan(),
                    "expected Istft barrier + MSE tail to activate whole-step fusion"
                );
            } else {
                no_allocations(|| tape_f.step_scalar(&params, rate).unwrap());
            }

            // FP64 step
            let s0_ref = ref_stft_f64(
                &x_ref, batch, length, fft_size, hop, padding, frames, &win_f64,
            );
            let mut s2_ref = vec![0.0f64; s0_ref.len()];
            for b in 0..batch {
                for f in 0..frames {
                    for k in 0..bins {
                        for c in 0..2 {
                            let idx = ((b * frames + f) * bins + k) * 2 + c;
                            let p = k * 2 + c;
                            s2_ref[idx] = s0_ref[idx] * m_ref[p] + b_ref[p];
                        }
                    }
                }
            }
            let y_ref = ref_istft_f64(
                &s2_ref, batch, length, fft_size, hop, padding, frames, &win_f64,
            );
            let scale = 2.0 / (batch * length) as f64;
            let dy_ref: Vec<f64> = y_ref
                .iter()
                .zip(&target_ref)
                .map(|(y, t)| scale * (y - t))
                .collect();
            let ds2_ref = ref_istft_backward_f64(
                &dy_ref, batch, length, fft_size, hop, padding, frames, &win_f64,
            );
            let mut dm_ref = vec![0.0f64; bins * 2];
            let mut db_ref = vec![0.0f64; bins * 2];
            let mut ds0_ref = vec![0.0f64; s0_ref.len()];
            for b in 0..batch {
                for f in 0..frames {
                    for k in 0..bins {
                        for c in 0..2 {
                            let idx = ((b * frames + f) * bins + k) * 2 + c;
                            let p = k * 2 + c;
                            db_ref[p] += ds2_ref[idx];
                            dm_ref[p] += ds2_ref[idx] * s0_ref[idx];
                            ds0_ref[idx] = ds2_ref[idx] * m_ref[p];
                        }
                    }
                }
            }
            let dx_ref = ref_stft_backward_f64(
                &ds0_ref, batch, length, fft_size, hop, padding, frames, &win_f64,
            );
            for (p, g) in x_ref.iter_mut().zip(&dx_ref) {
                *p -= rate as f64 * g;
            }
            for (p, g) in m_ref.iter_mut().zip(&dm_ref) {
                *p -= rate as f64 * g;
            }
            for (p, g) in b_ref.iter_mut().zip(&db_ref) {
                *p -= rate as f64 * g;
            }
        }

        close(x_u.read()?.data(), &x_ref, 2e-5, 2e-5);
        close(m_u.read()?.data(), &m_ref, 2e-5, 2e-5);
        close(b_u.read()?.data(), &b_ref, 2e-5, 2e-5);
        close_f32(x_f.read()?.data(), x_u.read()?.data(), 2e-5, 2e-5);
        close_f32(m_f.read()?.data(), m_u.read()?.data(), 2e-5, 2e-5);
        close_f32(b_f.read()?.data(), b_u.read()?.data(), 2e-5, 2e-5);

        // Also test uncentered Builder::stft -> Square -> Mean tape.
        let mut b2 = Builder::new()?;
        let x2 = b2.leaf([batch, length], true)?;
        let sp2 = b2.stft(x2, fft_size, hop, None)?;
        let sq2 = b2.unary(UnaryOp::Square, sp2)?;
        let loss2 = b2.reduce(ReduceOp::Mean, sq2, &[0, 1, 2, 3], false)?;
        let prog2 = b2.finish(&[loss2])?;
        let x2_u = Shared::new(Tensor::from_vec([batch, length], x_init.clone())?)?;
        let x2_f = Shared::new(Tensor::from_vec([batch, length], x_init)?)?;
        let mut t2_u = Tape::new(cpu, &prog2, &[Binding::new(x2, x2_u.clone())])?;
        let mut t2_f = Tape::new(cpu, &prog2, &[Binding::new(x2, x2_f.clone())])?;
        t2_u.forward()?;
        t2_u.backward_scalar(GradientMode::Overwrite)?;
        t2_u.sgd(&[x2], 0.01)?;
        t2_f.step_scalar(&[x2], 0.01)?;
        assert!(t2_f.has_fused_plan());
        close_f32(x2_f.read()?.data(), x2_u.read()?.data(), 2e-5, 2e-5);
    }
    Ok(())
}

#[test]
fn stft_backward_miri_strided_and_tape() -> fast_audio::Result<()> {
    let cpu = Cpu::Reference;
    let (batch, length, fft_size, hop) = (2, 6, 4, 2);
    let spec = StftSpec {
        batch,
        input_len: length,
        fft_size,
        hop,
    };
    let cspec = spec.centered()?;
    let x_rev = Layout::contiguous([batch, length])?.slice(1, length - 1, length, -1)?;
    let s_perm =
        Layout::contiguous([batch, cspec.bins(), cspec.frames(), 2])?.permute(&[0, 2, 1, 3])?;

    let x_t = Tensor::from_vec([batch, length], signal(batch * length, 901))?;
    let mut s_t = Tensor::zeros([batch, cspec.bins(), cspec.frames(), 2])?;
    let mut y_t = Tensor::zeros([batch, length])?;
    let mut dx_t = Tensor::zeros([batch, length])?;
    let mut ds_t = Tensor::zeros([batch, cspec.bins(), cspec.frames(), 2])?;

    let mut cfwd = CenteredStftPlan::new(cpu, cspec, &x_rev, &s_perm)?;
    let mut cbwd = CenteredStftBackward::new(cpu, cspec, &s_perm, &x_rev)?;
    let mut ifwd = IstftPlan::new(cpu, cspec, &s_perm, &x_rev)?;
    let mut ibwd = IstftBackward::new(cpu, cspec, &x_rev, &s_perm)?;

    cfwd.run(&x_t.view(&x_rev)?, &mut s_t.view_mut(&s_perm)?)?;
    ifwd.run(&s_t.view(&s_perm)?, &mut y_t.view_mut(&x_rev)?)?;
    close_f32(
        y_t.view(&x_rev)?.to_tensor()?.data(),
        x_t.view(&x_rev)?.to_tensor()?.data(),
        2e-5,
        2e-5,
    );
    ibwd.run(
        &x_t.view(&x_rev)?,
        &mut ds_t.view_mut(&s_perm)?,
        GradientMode::Overwrite,
    )?;
    cbwd.run(
        &ds_t.view(&s_perm)?,
        &mut dx_t.view_mut(&x_rev)?,
        GradientMode::Accumulate,
    )?;

    // Fused Tape step under Miri.
    let mut b = Builder::new()?;
    let lx = b.leaf([batch, length], true)?;
    let sp = b.centered_stft(lx, fft_size, hop, None)?;
    let y = b.istft(sp, length, fft_size, hop, None)?;
    let sq = b.unary(UnaryOp::Square, y)?;
    let loss = b.reduce(ReduceOp::Mean, sq, &[0, 1], false)?;
    let prog = b.finish(&[loss])?;
    let owner = Shared::new(x_t)?;
    let mut tape = Tape::new(cpu, &prog, &[Binding::new(lx, owner.clone())])?;
    tape.step_scalar(&[lx], 0.05)?;
    assert!(tape.has_fused_plan());
    tape.step_scalar(&[lx], 0.05)?;
    assert!(owner.read()?.data().iter().all(|v| v.is_finite()));
    Ok(())
}

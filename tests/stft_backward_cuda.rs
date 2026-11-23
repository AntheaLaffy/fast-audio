#![cfg(feature = "cuda")]

#[allow(dead_code)]
mod support;

use fast_audio::{
    BinaryOp, GradientMode, Layout, ReduceOp, Shared, StftSpec, Tensor, UnaryOp,
    autograd::{Binding, Builder},
    cpu::{self, Cpu},
    cuda::{
        self, CenteredStftBackward, CenteredStftPlan, Cuda, IstftBackward, IstftPlan, StftBackward,
        StftPlan,
    },
};
use support::{close, no_allocations};

fn signal(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed ^ 0x9e37_79b9_7f4a_7c15;
    (0..n)
        .map(|i| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((s >> 33) as f64) / ((1u64 << 31) as f64) - 0.5;
            (u * 1.3 + 0.2 * (i as f64 * 0.41).cos()) as f32
        })
        .collect()
}

fn custom_window(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let phase = i as f64 / n as f64;
            (0.65 - 0.35 * (std::f64::consts::TAU * phase).cos()
                + 0.16 * (std::f64::consts::TAU * phase).sin()) as f32
        })
        .collect()
}

fn close_f32(actual: &[f32], expected: &[f32], atol: f64, rtol: f64) {
    let exp: Vec<f64> = expected.iter().map(|&v| v as f64).collect();
    close(actual, &exp, atol, rtol);
}

#[test]
fn cuda_stft_centered_and_istft_forward_backward_match_cpu_and_strided() -> fast_audio::Result<()> {
    let cuda = Cuda::new(0)?;
    let other = Cuda::new(0)?;
    let cpu = Cpu::Reference;

    // 1. Uncentered STFT (even N=16, odd N=9, custom window, contiguous + odd-offset view)
    for (ci, &(batch, input_len, fft_size, hop, odd_offset)) in
        [(2, 48, 16, 8, false), (2, 31, 9, 4, true)]
            .iter()
            .enumerate()
    {
        let spec = StftSpec {
            batch,
            input_len,
            fft_size,
            hop,
        };
        let frames = spec.frames()?;
        let bins = spec.bins();
        let win = Tensor::from_vec([fft_size], custom_window(fft_size))?;
        let x_lay = Layout::contiguous([batch, input_len])?;
        let y_storage_len = batch * frames * bins * 2 + usize::from(odd_offset);
        let y_lay = if odd_offset {
            // Contiguous with odd f32 offset (tests 8-byte direct_r2c alignment guard).
            Layout::from_strides(
                [batch, frames, bins, 2],
                [(frames * bins * 2) as isize, (bins * 2) as isize, 2, 1],
                1,
            )?
        } else {
            Layout::contiguous([batch, frames, bins, 2])?
        };

        let x_h = Tensor::from_vec(
            [batch, input_len],
            signal(batch * input_len, 10 + ci as u64),
        )?;
        let dy_h = Tensor::from_vec([y_storage_len], signal(y_storage_len, 20 + ci as u64))?;
        let mut y_cpu = Tensor::zeros([y_storage_len])?;
        let mut dx_cpu = Tensor::zeros([batch, input_len])?;

        let mut c_fwd = cpu::StftPlan::with_window(cpu, spec, &win, &x_lay, &y_lay)?;
        let mut c_bwd = cpu::StftBackward::with_window(cpu, spec, &win, &y_lay, &x_lay)?;
        c_fwd.run(&x_h.view(&x_lay)?, &mut y_cpu.view_mut(&y_lay)?)?;
        c_bwd.run(
            &dy_h.view(&y_lay)?,
            &mut dx_cpu.view_mut(&x_lay)?,
            GradientMode::Overwrite,
        )?;

        let x_d = cuda.upload(&x_h)?;
        let dy_d = cuda.upload(&dy_h)?;
        let mut y_d = cuda.zeros([y_storage_len])?;
        let mut dx_d = cuda.zeros([batch, input_len])?;
        let mut g_fwd = StftPlan::with_window(&cuda, spec, &win, &x_lay, &y_lay)?;
        let mut g_bwd = StftBackward::with_window(&cuda, spec, &win, &y_lay, &x_lay)?;
        assert!(!g_bwd.input_required());
        assert_eq!(g_fwd.shared_memory_bytes_per_block(), 0);
        assert_eq!(g_bwd.shared_memory_bytes_per_block(), 0);

        g_fwd.run(&x_d.view(&x_lay)?, &mut y_d.view_mut(&y_lay)?)?;
        g_bwd.run(
            &dy_d.view(&y_lay)?,
            &mut dx_d.view_mut(&x_lay)?,
            GradientMode::Overwrite,
        )?;
        no_allocations(|| {
            g_fwd
                .run(
                    &x_d.view(&x_lay).unwrap(),
                    &mut y_d.view_mut(&y_lay).unwrap(),
                )
                .unwrap();
            g_bwd
                .run(
                    &dy_d.view(&y_lay).unwrap(),
                    &mut dx_d.view_mut(&x_lay).unwrap(),
                    GradientMode::Overwrite,
                )
                .unwrap();
        });
        cuda.synchronize()?;
        close_f32(
            cuda.download(&y_d)?.view(&y_lay)?.to_tensor()?.data(),
            y_cpu.view(&y_lay)?.to_tensor()?.data(),
            2e-5,
            2e-5,
        );
        close_f32(cuda.download(&dx_d)?.data(), dx_cpu.data(), 2e-5, 2e-5);

        // Foreign session rejection.
        let foreign_x = other.upload(&x_h)?;
        assert!(
            g_fwd
                .run(&foreign_x.view(&x_lay)?, &mut y_d.view_mut(&y_lay)?)
                .is_err()
        );
    }

    // 2. Centered STFT + iSTFT (including L = P + 1 minimal overlap, odd N, strided views, Accumulate)
    for (ci, &(batch, length, fft_size, hop, strided)) in [
        (2, 32, 16, 8, false),
        (2, 5, 8, 2, true),
        (2, 19, 9, 4, true),
    ]
    .iter()
    .enumerate()
    {
        let spec = StftSpec {
            batch,
            input_len: length,
            fft_size,
            hop,
        }
        .centered()?;
        let frames = spec.frames();
        let bins = spec.bins();
        let win = Tensor::from_vec([fft_size], custom_window(fft_size))?;

        let (x_lay, s_lay, s_store_shape) = if strided {
            (
                Layout::contiguous([batch, length])?.slice(1, length - 1, length, -1)?,
                Layout::contiguous([batch, bins, frames, 2])?.permute(&[0, 2, 1, 3])?,
                [batch, bins, frames, 2],
            )
        } else {
            (
                Layout::contiguous([batch, length])?,
                Layout::contiguous([batch, frames, bins, 2])?,
                [batch, frames, bins, 2],
            )
        };

        let x_h = Tensor::from_vec([batch, length], signal(batch * length, 100 + ci as u64))?;
        let mut s_in_vec = signal(batch * frames * bins * 2, 120 + ci as u64);
        s_in_vec[1] = 11.5;
        let s_in_h = Tensor::from_vec(s_store_shape, s_in_vec)?;
        let dy_h = Tensor::from_vec([batch, length], signal(batch * length, 140 + ci as u64))?;
        let init_dx_h = Tensor::from_vec([batch, length], vec![0.125f32; batch * length])?;
        let init_ds_h = Tensor::from_vec(s_store_shape, vec![-0.25f32; batch * frames * bins * 2])?;

        // CPU reference
        let mut s_out_cpu = Tensor::zeros(s_store_shape)?;
        let mut y_out_cpu = Tensor::zeros([batch, length])?;
        let mut dx_cpu = init_dx_h.clone();
        let mut ds_cpu = init_ds_h.clone();
        cpu::CenteredStftPlan::with_window(cpu, spec, &win, &x_lay, &s_lay)?
            .run(&x_h.view(&x_lay)?, &mut s_out_cpu.view_mut(&s_lay)?)?;
        cpu::CenteredStftBackward::with_window(cpu, spec, &win, &s_lay, &x_lay)?.run(
            &s_in_h.view(&s_lay)?,
            &mut dx_cpu.view_mut(&x_lay)?,
            GradientMode::Accumulate,
        )?;
        cpu::IstftPlan::with_window(cpu, spec, &win, &s_lay, &x_lay)?
            .run(&s_in_h.view(&s_lay)?, &mut y_out_cpu.view_mut(&x_lay)?)?;
        cpu::IstftBackward::with_window(cpu, spec, &win, &x_lay, &s_lay)?.run(
            &dy_h.view(&x_lay)?,
            &mut ds_cpu.view_mut(&s_lay)?,
            GradientMode::Accumulate,
        )?;

        // CUDA plans
        let x_d = cuda.upload(&x_h)?;
        let s_in_d = cuda.upload(&s_in_h)?;
        let dy_d = cuda.upload(&dy_h)?;
        let mut s_out_d = cuda.zeros(s_store_shape)?;
        let mut y_out_d = cuda.zeros([batch, length])?;
        let mut dx_d = cuda.upload(&init_dx_h)?;
        let mut ds_d = cuda.upload(&init_ds_h)?;

        let mut cfwd = CenteredStftPlan::with_window(&cuda, spec, &win, &x_lay, &s_lay)?;
        let mut cbwd = CenteredStftBackward::with_window(&cuda, spec, &win, &s_lay, &x_lay)?;
        let mut ifwd = IstftPlan::with_window(&cuda, spec, &win, &s_lay, &x_lay)?;
        let mut ibwd = IstftBackward::with_window(&cuda, spec, &win, &x_lay, &s_lay)?;

        cfwd.run(&x_d.view(&x_lay)?, &mut s_out_d.view_mut(&s_lay)?)?;
        cbwd.run(
            &s_in_d.view(&s_lay)?,
            &mut dx_d.view_mut(&x_lay)?,
            GradientMode::Accumulate,
        )?;
        ifwd.run(&s_in_d.view(&s_lay)?, &mut y_out_d.view_mut(&x_lay)?)?;
        ibwd.run(
            &dy_d.view(&x_lay)?,
            &mut ds_d.view_mut(&s_lay)?,
            GradientMode::Accumulate,
        )?;
        cuda.synchronize()?;

        close_f32(
            cuda.download(&s_out_d)?.data(),
            s_out_cpu.data(),
            2e-5,
            2e-5,
        );
        close_f32(cuda.download(&dx_d)?.data(), dx_cpu.data(), 2e-5, 2e-5);
        close_f32(
            cuda.download(&y_out_d)?.data(),
            y_out_cpu.data(),
            2e-5,
            2e-5,
        );
        close_f32(cuda.download(&ds_d)?.data(), ds_cpu.data(), 2e-5, 2e-5);

        // Also test IstftBackward Overwrite (exercises direct_r2c + in-place scale when contiguous).
        let mut ds_ov_cpu = Tensor::zeros(s_store_shape)?;
        let mut ds_ov_d = cuda.zeros(s_store_shape)?;
        cpu::IstftBackward::with_window(cpu, spec, &win, &x_lay, &s_lay)?.run(
            &dy_h.view(&x_lay)?,
            &mut ds_ov_cpu.view_mut(&s_lay)?,
            GradientMode::Overwrite,
        )?;
        ibwd.run(
            &dy_d.view(&x_lay)?,
            &mut ds_ov_d.view_mut(&s_lay)?,
            GradientMode::Overwrite,
        )?;
        cuda.synchronize()?;
        close_f32(
            cuda.download(&ds_ov_d)?.data(),
            ds_ov_cpu.data(),
            2e-5,
            2e-5,
        );
    }
    Ok(())
}

#[test]
fn cuda_spectral_tape_fused_and_captured_graph_matches_cpu_and_uncaptured() -> fast_audio::Result<()>
{
    let cuda = Cuda::new(0)?;
    let (batch, length, fft_size, hop) = (2, 64, 16, 8);
    let bins = fft_size / 2 + 1;
    let win = Tensor::from_vec([fft_size], custom_window(fft_size))?;

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

    let x_init = Tensor::from_vec([batch, length], signal(batch * length, 501))?;
    let mask_init = Tensor::from_vec([1, 1, bins, 2], signal(bins * 2, 502))?;
    let bias_init = Tensor::from_vec([1, 1, bins, 2], signal(bins * 2, 503))?;
    let target_init = Tensor::from_vec([batch, length], signal(batch * length, 504))?;

    // 1. CPU reference tape
    let x_c = Shared::new(x_init.clone())?;
    let m_c = Shared::new(mask_init.clone())?;
    let b_c = Shared::new(bias_init.clone())?;
    let t_c = Shared::new(target_init.clone())?;
    let mut cpu_tape = cpu::Tape::new(
        Cpu::Reference,
        &program,
        &[
            Binding::new(x, x_c.clone()),
            Binding::new(mask, m_c.clone()),
            Binding::new(bias, b_c.clone()),
            Binding::new(target, t_c.clone()),
        ],
    )?;

    // 2. CUDA uncaptured fused tape
    let x_u = Shared::new(cuda.upload(&x_init)?)?;
    let m_u = Shared::new(cuda.upload(&mask_init)?)?;
    let b_u = Shared::new(cuda.upload(&bias_init)?)?;
    let t_u = Shared::new(cuda.upload(&target_init)?)?;
    let mut gpu_uncap = cuda::Tape::new(
        &cuda,
        &program,
        &[
            Binding::new(x, x_u.clone()),
            Binding::new(mask, m_u.clone()),
            Binding::new(bias, b_u.clone()),
            Binding::new(target, t_u.clone()),
        ],
    )?;

    // 3. CUDA Graph captured fused tape (captures cuFFT R2C + cuFFT C2R + NVRTC fused tail + SGD!)
    let x_g = Shared::new(cuda.upload(&x_init)?)?;
    let m_g = Shared::new(cuda.upload(&mask_init)?)?;
    let b_g = Shared::new(cuda.upload(&bias_init)?)?;
    let t_g = Shared::new(cuda.upload(&target_init)?)?;
    let mut gpu_cg = cuda::Tape::new(
        &cuda,
        &program,
        &[
            Binding::new(x, x_g.clone()),
            Binding::new(mask, m_g.clone()),
            Binding::new(bias, b_g.clone()),
            Binding::new(target, t_g.clone()),
        ],
    )?;

    let params = [x, mask, bias];
    let rate = 0.05f32;
    for step in 0..6 {
        cpu_tape.step_scalar(&params, rate)?;
        gpu_uncap.step_scalar(&params, rate)?;
        if step == 0 {
            gpu_cg.step_scalar_captured(&params, rate)?;
            assert!(
                gpu_uncap.has_fused_plan(),
                "expected NVRTC fusion on post-iSTFT tail"
            );
            assert!(
                gpu_cg.has_fused_plan(),
                "expected NVRTC fusion in captured spectral tape"
            );
        } else if step == 1 {
            gpu_cg.step_scalar_captured(&params, rate)?;
            assert!(
                gpu_cg.has_captured_graph(),
                "expected CUDA Graph capture with cuFFT"
            );
        } else {
            no_allocations(|| gpu_cg.step_scalar_captured(&params, rate).unwrap());
        }
    }
    cuda.synchronize()?;

    // Bit-exact between uncaptured fused CUDA and CUDA Graph captured fused replay!
    assert_eq!(
        cuda.download(&*x_g.read()?)?.data(),
        cuda.download(&*x_u.read()?)?.data()
    );
    assert_eq!(
        cuda.download(&*m_g.read()?)?.data(),
        cuda.download(&*m_u.read()?)?.data()
    );
    assert_eq!(
        cuda.download(&*b_g.read()?)?.data(),
        cuda.download(&*b_u.read()?)?.data()
    );

    // Strict FP32 agreement vs CPU reference tape.
    close_f32(
        cuda.download(&*x_g.read()?)?.data(),
        x_c.read()?.data(),
        2e-5,
        2e-5,
    );
    close_f32(
        cuda.download(&*m_g.read()?)?.data(),
        m_c.read()?.data(),
        2e-5,
        2e-5,
    );
    close_f32(
        cuda.download(&*b_g.read()?)?.data(),
        b_c.read()?.data(),
        2e-5,
        2e-5,
    );
    Ok(())
}

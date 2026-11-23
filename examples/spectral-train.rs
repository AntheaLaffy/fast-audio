use fast_audio::{
    BinaryOp, ReduceOp, Result, Shared, Tensor, UnaryOp,
    autograd::{Binding, Builder, Program, Value},
    cpu::{self, Cpu},
};
use std::{env, f64::consts::TAU, time::Instant};

fn signal(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed ^ 0x9e37_79b9_7f4a_7c15;
    (0..n)
        .map(|i| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((s >> 33) as f64) / ((1u64 << 31) as f64) - 0.5;
            (u * 1.1 + 0.2 * (i as f64 * 0.13).sin()) as f32
        })
        .collect()
}

fn custom_window(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let phase = i as f64 / n as f64;
            (0.5 - 0.5 * (TAU * phase).cos()).sqrt() as f32
        })
        .collect()
}

struct Graph {
    program: Program,
    x: Value,
    mask: Value,
    bias: Value,
    target: Value,
}

fn build_graph(batch: usize, length: usize, fft_size: usize, hop: usize) -> Result<Graph> {
    let bins = fft_size / 2 + 1;
    let win = Tensor::from_vec([fft_size], custom_window(fft_size))?;
    let mut b = Builder::new()?;
    let x = b.leaf([batch, length], true)?;
    let mask = b.leaf([1, 1, bins, 2], true)?;
    let bias = b.leaf([1, 1, bins, 2], true)?;
    let target = b.leaf([batch, length], false)?;
    let s0 = b.centered_stft(x, fft_size, hop, Some(&win))?;
    let s1 = b.binary(BinaryOp::Mul, s0, mask)?;
    let s2 = b.binary(BinaryOp::Add, s1, bias)?;
    let y = b.istft(s2, length, fft_size, hop, Some(&win))?;
    let diff = b.binary(BinaryOp::Sub, y, target)?;
    let sq = b.unary(UnaryOp::Square, diff)?;
    let loss = b.reduce(ReduceOp::Mean, sq, &[0, 1], false)?;
    let program = b.finish(&[loss])?;
    Ok(Graph {
        program,
        x,
        mask,
        bias,
        target,
    })
}

fn run_cpu(
    batch: usize,
    length: usize,
    fft_size: usize,
    hop: usize,
    warmup: usize,
    iters: usize,
    rate: f32,
) -> Result<()> {
    let g = build_graph(batch, length, fft_size, hop)?;
    let bins = fft_size / 2 + 1;
    let x_init = Tensor::from_vec([batch, length], signal(batch * length, 101))?;
    let mask_init = Tensor::from_vec([1, 1, bins, 2], signal(bins * 2, 102))?;
    let bias_init = Tensor::from_vec([1, 1, bins, 2], signal(bins * 2, 103))?;
    let target_init = Tensor::from_vec([batch, length], signal(batch * length, 104))?;

    #[cfg(feature = "mkl")]
    let cpu = Cpu::mkl(1)?;
    #[cfg(not(feature = "mkl"))]
    let cpu = Cpu::Reference;

    let x_o = Shared::new(x_init.clone())?;
    let m_o = Shared::new(mask_init.clone())?;
    let b_o = Shared::new(bias_init.clone())?;
    let t_o = Shared::new(target_init)?;
    let mut tape = cpu::Tape::new(
        cpu,
        &g.program,
        &[
            Binding::new(g.x, x_o.clone()),
            Binding::new(g.mask, m_o.clone()),
            Binding::new(g.bias, b_o.clone()),
            Binding::new(g.target, t_o),
        ],
    )?;
    let params = [g.x, g.mask, g.bias];
    // Run exact 4-step numerical snapshot first on fresh copies.
    for _ in 0..4 {
        tape.step_scalar(&params, rate)?;
    }
    let check_x: Vec<f32> = x_o.read()?.data().iter().take(16).copied().collect();
    let check_m: Vec<f32> = m_o.read()?.data().iter().take(16).copied().collect();
    let check_b: Vec<f32> = b_o.read()?.data().iter().take(16).copied().collect();

    // Reset and time warmed iterations.
    x_o.write()?.data_mut().copy_from_slice(x_init.data());
    m_o.write()?.data_mut().copy_from_slice(mask_init.data());
    b_o.write()?.data_mut().copy_from_slice(bias_init.data());
    for _ in 0..warmup {
        tape.step_scalar(&params, rate)?;
    }
    let t0 = Instant::now();
    for _ in 0..iters {
        tape.step_scalar(&params, rate)?;
    }
    let wall_us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
    println!(
        "{{\"backend\":\"cpu\",\"fused\":{},\"wall_us\":{:.4},\"x\":{:?},\"m\":{:?},\"b\":{:?}}}",
        tape.has_fused_plan(),
        wall_us,
        check_x,
        check_m,
        check_b
    );
    Ok(())
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn run_cuda(
    batch: usize,
    length: usize,
    fft_size: usize,
    hop: usize,
    warmup: usize,
    iters: usize,
    rate: f32,
    captured: bool,
) -> Result<()> {
    use fast_audio::cuda::{self, Cuda};
    let cuda = Cuda::new(0)?;
    let g = build_graph(batch, length, fft_size, hop)?;
    let bins = fft_size / 2 + 1;
    let x_init = Tensor::from_vec([batch, length], signal(batch * length, 101))?;
    let mask_init = Tensor::from_vec([1, 1, bins, 2], signal(bins * 2, 102))?;
    let bias_init = Tensor::from_vec([1, 1, bins, 2], signal(bins * 2, 103))?;
    let target_init = Tensor::from_vec([batch, length], signal(batch * length, 104))?;

    let x_o = Shared::new(cuda.upload(&x_init)?)?;
    let m_o = Shared::new(cuda.upload(&mask_init)?)?;
    let b_o = Shared::new(cuda.upload(&bias_init)?)?;
    let t_o = Shared::new(cuda.upload(&target_init)?)?;
    let mut tape = cuda::Tape::new(
        &cuda,
        &g.program,
        &[
            Binding::new(g.x, x_o.clone()),
            Binding::new(g.mask, m_o.clone()),
            Binding::new(g.bias, b_o.clone()),
            Binding::new(g.target, t_o),
        ],
    )?;
    let params = [g.x, g.mask, g.bias];
    for _ in 0..4 {
        if captured {
            tape.step_scalar_captured(&params, rate)?;
        } else {
            tape.step_scalar(&params, rate)?;
        }
    }
    cuda.synchronize()?;
    let check_x: Vec<f32> = cuda
        .download(&*x_o.read()?)?
        .data()
        .iter()
        .take(16)
        .copied()
        .collect();
    let check_m: Vec<f32> = cuda
        .download(&*m_o.read()?)?
        .data()
        .iter()
        .take(16)
        .copied()
        .collect();
    let check_b: Vec<f32> = cuda
        .download(&*b_o.read()?)?
        .data()
        .iter()
        .take(16)
        .copied()
        .collect();

    cuda.copy_from_host(&x_init, &mut *x_o.write()?)?;
    cuda.copy_from_host(&mask_init, &mut *m_o.write()?)?;
    cuda.copy_from_host(&bias_init, &mut *b_o.write()?)?;
    for _ in 0..warmup {
        if captured {
            tape.step_scalar_captured(&params, rate)?;
        } else {
            tape.step_scalar(&params, rate)?;
        }
    }
    let timing = cuda.measure(iters, || {
        if captured {
            tape.step_scalar_captured(&params, rate)
        } else {
            tape.step_scalar(&params, rate)
        }
    })?;
    println!(
        "{{\"backend\":\"{}\",\"fused\":{},\"captured\":{},\"wall_us\":{:.4},\"device_us\":{:.4},\"x\":{:?},\"m\":{:?},\"b\":{:?}}}",
        if captured { "cuda-cg" } else { "cuda" },
        tape.has_fused_plan(),
        tape.has_captured_graph(),
        timing.wall_us,
        timing.device_us,
        check_x,
        check_m,
        check_b
    );
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    let mode = args.get(1).map_or("cpu", String::as_str);
    let batch: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(2);
    let length: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(16000);
    let fft_size: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(512);
    let hop: usize = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(128);
    let warmup: usize = args.get(6).and_then(|s| s.parse().ok()).unwrap_or(20);
    let iters: usize = args.get(7).and_then(|s| s.parse().ok()).unwrap_or(200);
    let rate = 0.01f32;
    match mode {
        "cpu" => run_cpu(batch, length, fft_size, hop, warmup, iters, rate),
        #[cfg(feature = "cuda")]
        "cuda" => run_cuda(batch, length, fft_size, hop, warmup, iters, rate, false),
        #[cfg(feature = "cuda")]
        "cuda-cg" => run_cuda(batch, length, fft_size, hop, warmup, iters, rate, true),
        _ => panic!("unknown mode: {mode}"),
    }
}

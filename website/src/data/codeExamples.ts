export interface CodeExample {
  id: string;
  tabEn: string;
  tabZh: string;
  filename: string;
  taglineEn: string;
  taglineZh: string;
  badges: string[];
  code: string;
}

export const CODE_EXAMPLES: CodeExample[] = [
  {
    id: 'first-tensor',
    tabEn: 'First tensor',
    tabZh: '第一个张量',
    filename: 'main.rs',
    taglineEn: 'Multiply two tensors on the portable CPU backend and check both output samples.',
    taglineZh: '在普通 CPU 上运行张量乘法，并检查两个输出值。',
    badges: ['Portable CPU'],
    code: `use fast_audio::{Tensor, cpu::Cpu};

fn main() -> Result<(), fast_audio::Error> {
    let cpu = Cpu::Reference; // Portable CPU backend
    let a = Tensor::from_vec([2, 3], vec![1., 2., 3., 4., 5., 6.])?;
    let b = Tensor::from_vec([3, 1], vec![1., 0., -1.])?;
    let mut output = Tensor::zeros([2, 1])?;
    cpu.matmul(&a, &b, &mut output)?;
    assert_eq!(output.data(), &[-2., -2.]);
    Ok(())
}`,
  },
  {
    id: 'fusion-tape',
    tabEn: 'Training tape',
    tabZh: '训练 Tape',
    filename: 'training_step.rs',
    taglineEn:
      'Prepare a waveform training graph once, then reuse its tape for each SGD step on the portable CPU.',
    taglineZh:
      '在普通 CPU 上构建波形训练图，然后在每个 SGD 步骤中复用同一份 Tape。',
    badges: ['Portable CPU', 'Strict FP32'],
    code: `use fast_audio::{
    BinaryOp, ReduceOp, Result, Shared, Tensor, UnaryOp,
    autograd::{Binding, Builder},
    cpu::{self, Cpu},
};

pub fn train_waveform_block() -> Result<()> {
    // 1. Declare a checked, fixed-shape reverse-mode DAG
    let mut b = Builder::new()?;
    let x      = b.leaf([4, 1024], false)?;
    let gain   = b.leaf([4, 1], true)?;
    let bias   = b.leaf([4, 1], true)?;
    let target = b.leaf([4, 1024], false)?;

    let scaled = b.binary(BinaryOp::Mul, x, gain)?;
    let shifted = b.binary(BinaryOp::Add, scaled, bias)?;
    let pred   = b.unary(UnaryOp::Tanh, shifted)?;
    let diff   = b.binary(BinaryOp::Sub, pred, target)?;
    let sq     = b.unary(UnaryOp::Square, diff)?;
    let loss   = b.reduce(ReduceOp::Mean, sq, &[0, 1], false)?;
    let program = b.finish(&[loss])?;

    // 2. Bind guarded Shared<Tensor> owners — prepare the reusable tape
    let cpu = Cpu::Reference;
    let g_own = Shared::new(Tensor::from_vec([4, 1], vec![1.; 4])?)?;
    let b_own = Shared::new(Tensor::zeros([4, 1])?)?;
    let mut tape = cpu::Tape::new(cpu, &program, &[
        Binding::new(x,      Shared::new(Tensor::zeros([4, 1024])?)?),
        Binding::new(gain,   g_own.clone()),
        Binding::new(bias,   b_own.clone()),
        Binding::new(target, Shared::new(Tensor::zeros([4, 1024])?)?),
    ])?;

    for _ in 0..100 {
        tape.step_scalar(&[gain, bias], 1e-2)?; // Forward, backward and SGD
    }
    Ok(())
}`,
  },
  {
    id: 'cuda-graph',
    tabEn: 'CUDA Graph + NVRTC',
    tabZh: 'CUDA Graph 重放',
    filename: 'cuda_captured.rs',
    taglineEn:
      'Compile fused strict-FP32 PTX via NVRTC before capture, record the entire training step into a guarded CUDA Graph, and replay with zero host allocations.',
    taglineZh:
      '在捕获前经 NVRTC 编译严格 FP32 融合核，将整步训练直接录制进受版本号/指针保护的 CUDA Graph，热路径零分配重放。',
    badges: ['12/12 Sweep vs PyTorch', 'SHA-256 Bit-Exact', 'compute-sanitizer 0 err'],
    code: `use fast_audio::{
    Result, Shared, Tensor,
    autograd::{Binding, Program, Value},
    cuda::{self, Cuda},
};

pub fn run_captured_cuda_step(
    program: &Program,
    leaves: &[(Value, Tensor)],
    params: &[Value],
) -> Result<()> {
    // Owned single-session CUDA context with static 4 MiB cuBLAS workspace
    let gpu = Cuda::new(0)?;
    let mut bindings = Vec::with_capacity(leaves.len());
    for (val, host_tensor) in leaves {
        let dev = gpu.upload(host_tensor)?;
        bindings.push(Binding::new(*val, Shared::new(dev)?));
    }

    // NVRTC compiles strict-FP32 fused kernels (no --use_fast_math) into PTX cache
    let mut tape = cuda::Tape::new(&gpu, program, &bindings)?;

    // First call captures; subsequent calls validate pointers/generations & replay
    for _ in 0..1000 {
        tape.step_scalar_captured(params, 1e-2)?;
    }
    gpu.synchronize()?;
    Ok(())
}`,
  },
  {
    id: 'spectral-diff',
    tabEn: 'Differentiable STFT',
    tabZh: '可微频谱 STFT',
    filename: 'examples/spectral-train.rs',
    taglineEn:
      'End-to-end differentiable centered STFT → complex mask → NOLA-verified inverse STFT with zero-atomic GPU backward kernels and automatic tail fusion.',
    taglineZh:
      '端到端可微居中 STFT → 复数频谱掩码 → NOLA 校验逆 STFT，GPU 反向零原子操作，前后逐元素/损失尾部自动融合。',
    badges: ['In Workspace (v1)', 'Zero-Atomic GPU', 'NOLA FP64 Checked'],
    code: `use fast_audio::{
    BinaryOp, ReduceOp, Result, Tensor, UnaryOp,
    autograd::{Builder, Program},
};

pub fn build_spectral_mask_graph(
    batch: usize,
    length: usize,
    fft_size: usize,
    hop: usize,
    sqrt_hann: &Tensor,
) -> Result<Program> {
    let bins = fft_size / 2 + 1;
    let mut b = Builder::new()?;
    let x      = b.leaf([batch, length], true)?;
    let mask   = b.leaf([1, 1, bins, 2], true)?;
    let bias   = b.leaf([1, 1, bins, 2], true)?;
    let target = b.leaf([batch, length], false)?;

    // Linear spectral barriers: zero saved forward activations required for VJP
    let s0   = b.centered_stft(x, fft_size, hop, Some(sqrt_hann))?;
    let s1   = b.binary(BinaryOp::Mul, s0, mask)?;
    let s2   = b.binary(BinaryOp::Add, s1, bias)?;
    let y    = b.istft(s2, length, fft_size, hop, Some(sqrt_hann))?;

    // Post-iSTFT MSE loss + SGD automatically fuse into a single tiled/NVRTC tail
    let diff = b.binary(BinaryOp::Sub, y, target)?;
    let sq   = b.unary(UnaryOp::Square, diff)?;
    let loss = b.reduce(ReduceOp::Mean, sq, &[0, 1], false)?;
    b.finish(&[loss])
}`,
  },
  {
    id: 'views-versions',
    tabEn: 'Strided Views & Versions',
    tabZh: '步幅视图与版本守卫',
    filename: 'views_and_storage.rs',
    taglineEn:
      'Arbitrary-rank signed strides, zero-copy broadcast/slice/permute, compile-time borrow rules, and write-generation invalidation with zero overhead on inference.',
    taglineZh:
      '任意秩有符号步幅、零拷贝切片/转置/广播，结合编译期借用检查与运行时写版本号失效保护，对普通推理张量零元数据开销。',
    badges: ['Miri 0 UB', 'Injectivity Proof', 'Zero Copy'],
    code: `use fast_audio::{Layout, Result, Tensor, Tracked};

pub fn checked_strided_views() -> Result<()> {
    // Separate logical geometry from contiguous storage
    let base = Layout::contiguous([2, 3, 4])?;
    let transposed = base.permute(&[2, 0, 1])?;       // [4, 2, 3], zero copy
    let reversed   = transposed.slice(0, 3, 4, -1)?;  // Negative stride, bounds checked

    // Opt-in mutation version tracking (ordinary Tensor keeps 48-byte footprint)
    let mut tracked = Tracked::new(Tensor::zeros([2, 3, 4])?)?;
    let saved_ver = tracked.version();

    // Immutable read view preserves generation; overlapping broadcasts allowed
    {
        let read_view = tracked.read().view(&reversed)?;
        assert_eq!(read_view.layout().shape().dims(), &[4, 2, 3]);
    }

    // Exclusive write advances generation BEFORE exposing mutable access
    let _mut_view = tracked.write()?.view_mut(&reversed)?; // Requires injectivity proof
    assert!(tracked.check_version(saved_ver).is_err());     // Stale saved value rejected
    Ok(())
}`,
  },
];

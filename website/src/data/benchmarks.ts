export type WorkloadFamily = 'all' | 'dense' | 'waveform' | 'spectrum';

export interface BenchmarkCase {
  id: string;
  family: Exclude<WorkloadFamily, 'all'>;
  nativeUs: number;
  torchUs: number;
  ratio: number; // native / torch (< 1 means fast-audio wins)
  speedup: number; // torch / native (> 1 means fast-audio wins)
  bitExact?: boolean;
  noteEn?: string;
  noteZh?: string;
}

export interface EvolutionStage {
  commit: string;
  date: string;
  titleEn: string;
  titleZh: string;
  summaryEn: string;
  summaryZh: string;
  cpuHighlightEn: string;
  cpuHighlightZh: string;
  gpuHighlightEn: string;
  gpuHighlightZh: string;
  gpuGeomeanVsTorch: string;
  cpuGeomeanVsTorch: string;
  winsGpu: string;
  winsCpu: string;
}

export interface GtcrnCase {
  id: string;
  labelEn: string;
  labelZh: string;
  intelNativeMs: number;
  intelEagerMs: number;
  gpuNativeEagerCtxMs: number;
  gpuEagerMs: number;
  gpuNativeGraphCtxMs: number;
  gpuPreparedGraphMs: number;
}

// Verified 2026-11-20 paired 5-block benchmark on Colab NVIDIA L4 (SM 8.9, CUDA 13.0)
// Commit e6ad598 (NVRTC strict-FP32 whole-step fusion + CUDA Graph) vs PyTorch 2.14.1 compiled-cuda-graph
export const NVIDIA_L4_CASES: BenchmarkCase[] = [
  {
    id: 'dense-tiny',
    family: 'dense',
    nativeUs: 8.63,
    torchUs: 9.41,
    ratio: 0.918,
    speedup: 9.41 / 8.63,
    bitExact: true,
  },
  {
    id: 'dense-tail',
    family: 'dense',
    nativeUs: 14.25,
    torchUs: 14.29,
    ratio: 0.997,
    speedup: 14.29 / 14.25,
    bitExact: true,
  },
  {
    id: 'dense-projection',
    family: 'dense',
    nativeUs: 14.58,
    torchUs: 16.27,
    ratio: 0.896,
    speedup: 16.27 / 14.58,
    bitExact: true,
  },
  {
    id: 'dense-large',
    family: 'dense',
    nativeUs: 48.88,
    torchUs: 50.76,
    ratio: 0.963,
    speedup: 50.76 / 48.88,
    bitExact: true,
  },
  {
    id: 'wave-tiny',
    family: 'waveform',
    nativeUs: 2.56,
    torchUs: 4.10,
    ratio: 0.624,
    speedup: 4.10 / 2.56,
    bitExact: true,
  },
  {
    id: 'wave-block',
    family: 'waveform',
    nativeUs: 3.45,
    torchUs: 3.96,
    ratio: 0.870,
    speedup: 3.96 / 3.45,
    bitExact: true,
  },
  {
    id: 'wave-multichannel',
    family: 'waveform',
    nativeUs: 5.45,
    torchUs: 12.42,
    ratio: 0.439,
    speedup: 12.42 / 5.45,
    bitExact: true,
  },
  {
    id: 'wave-long',
    family: 'waveform',
    nativeUs: 5.15,
    torchUs: 11.39,
    ratio: 0.452,
    speedup: 11.39 / 5.15,
    bitExact: true,
  },
  {
    id: 'spectrum-tiny',
    family: 'spectrum',
    nativeUs: 2.56,
    torchUs: 5.11,
    ratio: 0.500,
    speedup: 5.11 / 2.56,
    bitExact: true,
  },
  {
    id: 'spectrum-block',
    family: 'spectrum',
    nativeUs: 4.08,
    torchUs: 5.75,
    ratio: 0.710,
    speedup: 5.75 / 4.08,
    bitExact: true,
  },
  {
    id: 'spectrum-medium',
    family: 'spectrum',
    nativeUs: 5.06,
    torchUs: 13.52,
    ratio: 0.374,
    speedup: 13.52 / 5.06,
    bitExact: true,
  },
  {
    id: 'spectrum-large',
    family: 'spectrum',
    nativeUs: 8.08,
    torchUs: 28.10,
    ratio: 0.287,
    speedup: 28.10 / 8.08,
    bitExact: true,
    noteEn: '35.3x faster than pre-fusion CUDA Graph',
    noteZh: '较融合前 CUDA Graph 提速 35.3 倍',
  },
];

// Verified 2026-11-20 paired 5-block benchmark on Colab Intel CPU (oneMKL, 1 thread)
// Commit e6ad598 (zero-unsafe L1-tiled fusion + MKL VML) vs PyTorch 2.14.1 torch.compile(inductor)
export const INTEL_CPU_CASES: BenchmarkCase[] = [
  {
    id: 'dense-tiny',
    family: 'dense',
    nativeUs: 1.38,
    torchUs: 101.32,
    ratio: 1.38 / 101.32,
    speedup: 73.5,
    bitExact: true,
  },
  {
    id: 'dense-tail',
    family: 'dense',
    nativeUs: 5.99,
    torchUs: 108.96,
    ratio: 5.99 / 108.96,
    speedup: 18.2,
    bitExact: true,
  },
  {
    id: 'dense-projection',
    family: 'dense',
    nativeUs: 575.57,
    torchUs: 733.12,
    ratio: 575.57 / 733.12,
    speedup: 1.27,
    bitExact: true,
  },
  {
    id: 'dense-large',
    family: 'dense',
    nativeUs: 5544.59,
    torchUs: 5891.29,
    ratio: 5544.59 / 5891.29,
    speedup: 1.06,
    bitExact: true,
  },
  {
    id: 'wave-tiny',
    family: 'waveform',
    nativeUs: 3.06,
    torchUs: 90.25,
    ratio: 3.06 / 90.25,
    speedup: 29.5,
  },
  {
    id: 'wave-block',
    family: 'waveform',
    nativeUs: 4.39,
    torchUs: 86.36,
    ratio: 4.39 / 86.36,
    speedup: 19.7,
  },
  {
    id: 'wave-multichannel',
    family: 'waveform',
    nativeUs: 513.68,
    torchUs: 749.36,
    ratio: 513.68 / 749.36,
    speedup: 1.46,
  },
  {
    id: 'wave-long',
    family: 'waveform',
    nativeUs: 618.08,
    torchUs: 721.74,
    ratio: 618.08 / 721.74,
    speedup: 1.17,
  },
  {
    id: 'spectrum-tiny',
    family: 'spectrum',
    nativeUs: 5.07,
    torchUs: 87.48,
    ratio: 5.07 / 87.48,
    speedup: 17.2,
  },
  {
    id: 'spectrum-block',
    family: 'spectrum',
    nativeUs: 35.12,
    torchUs: 100.66,
    ratio: 35.12 / 100.66,
    speedup: 2.87,
  },
  {
    id: 'spectrum-medium',
    family: 'spectrum',
    nativeUs: 1556.89,
    torchUs: 883.33,
    ratio: 1556.89 / 883.33,
    speedup: 883.33 / 1556.89,
    noteEn: 'Open gap vs Inductor SIMD codegen (1.58x over M3)',
    noteZh: '尚落后于 Inductor（较 M3 已提速 1.58x，持续优化中）',
  },
  {
    id: 'spectrum-large',
    family: 'spectrum',
    nativeUs: 8716.73,
    torchUs: 5586.42,
    ratio: 8716.73 / 5586.42,
    speedup: 5586.42 / 8716.73,
    noteEn: 'Open gap vs Inductor SIMD codegen (2.20x over M3)',
    noteZh: '尚落后于 Inductor（较 M3 已提速 2.20x，持续优化中）',
  },
];

export const EVOLUTION_STAGES: EvolutionStage[] = [
  {
    commit: '3b66357',
    date: '2026-10-20',
    titleEn: 'Stage 0 · First Native Training Baseline',
    titleZh: '阶段 0 · 首个原生训练步基线',
    summaryEn:
      'Complete forward + reverse-mode tape + SGD baseline verified against FP64/PyTorch. Exposed severe CPU coordinate-decoding overhead and GPU kernel-launch floor.',
    summaryZh:
      '打通首个前向 + 反向 Tape + SGD 完整训练步并通过 FP64/PyTorch 验证，暴露出 CPU 坐标解码与 GPU 多核启动开销等主要瓶颈。',
    cpuHighlightEn: 'Large spectral trailed Inductor by ~13.1x (58.5 ms vs 4.5 ms)',
    cpuHighlightZh: '大频谱落后 Inductor 约 13.1 倍（58.5 ms vs 4.5 ms）',
    gpuHighlightEn: 'Trailed PyTorch CUDA Graph on 12/12 cases (1.7x–11.7x slower)',
    gpuHighlightZh: '12/12 全线落后于 PyTorch CUDA Graph（慢 1.7x–11.7x）',
    gpuGeomeanVsTorch: '5.2x slower',
    cpuGeomeanVsTorch: '5/12 wins',
    winsGpu: '0 / 12',
    winsCpu: '5 / 12',
  },
  {
    commit: '04ce7d6',
    date: '2026-10-20',
    titleEn: 'Stage 1 · Prepared CPU Backward Dispatch',
    titleZh: '阶段 1 · 预备 CPU 反向分发',
    summaryEn:
      'Preselect operation/strategy kernels once at plan construction with an explicit +8-byte fixed header budget and zero dynamic workspace growth.',
    summaryZh:
      '在计划构建阶段一次性预选算子与策略核函数，固定头部仅增 8 字节，动态工作区与热态分配零增长。',
    cpuHighlightEn: '1.5648x confirmed primary speedup; 12/12 slowdown guards pass',
    cpuHighlightZh: '独立确认 1.5648x 主音频提速，12/12 退化红线全部通过',
    gpuHighlightEn: 'CPU-only targeted intervention',
    gpuHighlightZh: '本阶段仅针对 CPU 路径',
    gpuGeomeanVsTorch: '—',
    cpuGeomeanVsTorch: '1.56x native',
    winsGpu: '0 / 12',
    winsCpu: '5 / 12',
  },
  {
    commit: 'e98ddb1',
    date: '2026-11-20',
    titleEn: 'Stage 2 · Bit-Exact Reduction & Indexing',
    titleZh: '阶段 2 · 位精确小轴归约与寻址优化',
    summaryEn:
      'Specialized 2-element / sub-warp / zero-division CUDA kernels and chunked Tree256 CPU reductions with exact SHA-256 bit parity.',
    summaryZh:
      '引入 2 元素对齐加载、半 Warp 洗牌与无除法增量索引 CUDA 核及 CPU 分块 Tree256 归约，12/12 位级一致。',
    cpuHighlightEn: '+1.639x geomean speedup; spectrum cases jumped 3.16x–3.30x',
    cpuHighlightZh: '几何平均再提速 1.639x，频谱用例飙升 3.16x–3.30x',
    gpuHighlightEn: 'spectrum-large 523.05 → 316.46 us; beats Eager 12/12 & Compiled 11/12',
    gpuHighlightZh: 'spectrum-large 523.05 → 316.46 us；胜 Eager 12/12、Compiled 11/12',
    gpuGeomeanVsTorch: '11/12 vs Compiled',
    cpuGeomeanVsTorch: '0.346x (2.89x faster)',
    winsGpu: '11/12 Compiled',
    winsCpu: '6 / 12 Inductor',
  },
  {
    commit: 'b1ee58f',
    date: '2026-11-20',
    titleEn: 'Stage 3 · Intel MKL VML + Guarded CUDA Graph',
    titleZh: '阶段 3 · Intel MKL VML + 受保护 CUDA Graph 重放',
    summaryEn:
      'Added single-threaded high-accuracy oneMKL VML transcendentals, 8-block multi-SM split reductions, and version-checked zero-alloc CUDA Graph step replay.',
    summaryZh:
      '接入高精度 oneMKL VML 超越函数、多 SM 分裂归约，以及带版本号/指针守卫的零分配 CUDA Graph 整步捕获重放。',
    cpuHighlightEn: '+1.169x geomean; wave-long 2.10x faster; sweeps PyTorch Eager 12/12',
    cpuHighlightZh: '几何平均再提速 1.169x；wave-long 快 2.10x；12/12 横扫 PyTorch Eager',
    gpuHighlightEn: '+1.880x geomean via CUDA Graph (dense-tiny 49.81 → 14.84 us); 12/12 bit-exact',
    gpuHighlightZh: 'CUDA Graph 提速 1.880x（dense-tiny 49.81 → 14.84 us）；12/12 位一致',
    gpuGeomeanVsTorch: '3.67x gap left',
    cpuGeomeanVsTorch: '0.296x (3.38x faster)',
    winsGpu: '0 / 12 Graph',
    winsCpu: '8 / 12 Inductor',
  },
  {
    commit: 'e6ad598',
    date: '2026-11-20',
    titleEn: 'Stage 4 · Whole-Step Tape Fusion Compiler',
    titleZh: '阶段 4 · 通用全步 Tape 融合编译器',
    summaryEn:
      'General DAG fusion compiler lowering forward + backward + SGD into zero-unsafe L1-tiled CPU loops or NVRTC strict-FP32 CUDA kernels recorded directly inside CUDA Graphs.',
    summaryZh:
      '通用 DAG 融合编译器将前向 + 反向 + SGD 自动降级为 CPU 零 unsafe L1 分块循环，或由 NVRTC 编译为严格 FP32 融合核并直接录入 CUDA Graph。',
    cpuHighlightEn: '+1.368x geomean; 0.237x vs Inductor (4.22x faster geomean, 10/12 wins)',
    cpuHighlightZh: '再提速 1.368x；对比 Inductor 仅 0.237x（快 4.22x，胜 10/12）',
    gpuHighlightEn: '+10.963x geomean; 12/12 CLEAN SWEEP vs PyTorch CUDA Graph (0.620x / 1.61x faster)',
    gpuHighlightZh: '暴涨 10.963x；12/12 完胜 PyTorch CUDA Graph（仅 0.620x，快 1.61x）',
    gpuGeomeanVsTorch: '0.620x (1.61x faster)',
    cpuGeomeanVsTorch: '0.237x (4.22x faster)',
    winsGpu: '12 / 12 Sweep',
    winsCpu: '10 / 12 Inductor',
  },
];

// Verified GTCRN complete speech-enhancement model regression (commit a7caf75, 16 kHz, medians in ms)
export const GTCRN_CASES: GtcrnCase[] = [
  {
    id: 'b1-4096',
    labelEn: 'Batch 1 · 4,096 samples (0.26s)',
    labelZh: 'Batch 1 · 4,096 采样点 (0.26秒)',
    intelNativeMs: 8.436,
    intelEagerMs: 12.193,
    gpuNativeEagerCtxMs: 1.138,
    gpuEagerMs: 8.178,
    gpuNativeGraphCtxMs: 1.120,
    gpuPreparedGraphMs: 0.661,
  },
  {
    id: 'b1-16000',
    labelEn: 'Batch 1 · 16,000 samples (1.0s)',
    labelZh: 'Batch 1 · 16,000 采样点 (1.0秒)',
    intelNativeMs: 29.977,
    intelEagerMs: 26.351,
    gpuNativeEagerCtxMs: 1.748,
    gpuEagerMs: 8.283,
    gpuNativeGraphCtxMs: 1.718,
    gpuPreparedGraphMs: 0.878,
  },
  {
    id: 'b1-64000',
    labelEn: 'Batch 1 · 64,000 samples (4.0s)',
    labelZh: 'Batch 1 · 64,000 采样点 (4.0秒)',
    intelNativeMs: 121.786,
    intelEagerMs: 90.378,
    gpuNativeEagerCtxMs: 3.892,
    gpuEagerMs: 8.369,
    gpuNativeGraphCtxMs: 3.857,
    gpuPreparedGraphMs: 1.876,
  },
  {
    id: 'b2-16000',
    labelEn: 'Batch 2 · 16,000 samples (2x1.0s)',
    labelZh: 'Batch 2 · 16,000 采样点 (2x1.0秒)',
    intelNativeMs: 64.000,
    intelEagerMs: 37.433,
    gpuNativeEagerCtxMs: 2.270,
    gpuEagerMs: 8.727,
    gpuNativeGraphCtxMs: 2.228,
    gpuPreparedGraphMs: 1.063,
  },
];

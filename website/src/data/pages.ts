import { words, type Lang } from './i18n';

export const pages = {
  '': { title: words('Deep learning for audio', '音频深度学习', '音声のための深層学習'), desc: words('Audio training and inference in Rust.', '用 Rust 训练和运行音频模型。', 'Rust による音声モデルの学習と推論。'), terms: '' },
  'docs/install': { title: words('Install fast-audio', '安装 fast-audio', 'fast-audio のインストール'), desc: words('Choose a compute backend and add the Rust library to your project, or build from source.', '选择计算后端，将 Rust 库添加到项目中，也可以从源码构建。', '実行環境を選び、Rust プロジェクトへ追加するか、ソースからビルドします。'), terms: 'install setup cargo rust quick start CUDA cuDNN oneDNN MKL 安装 快速上手 インストール セットアップ' },
  tutorials: { title: words('Tutorials', '教程', 'チュートリアル'), desc: words('Work through a tensor operation, a spectral round trip, and a complete training step on the portable CPU backend.', '在通用 CPU 后端上完成一次张量运算、频谱重建和一个完整的训练步骤。', '標準 CPU バックエンドでテンソル演算、スペクトルからの波形復元、学習ステップを実行します。'), terms: 'examples guide 学习 示例 初心者 入門' },
  'tutorials/tensors': { title: words('Your first tensor', '第一个张量', 'はじめてのテンソル'), desc: words('Multiply two tensors and check the shape and values of the result.', '用两个张量做矩阵乘法，检查输出的形状和值。', '2 つのテンソルの行列積を求め、出力の形状と値を確かめます。'), terms:'Tensor Cpu::Reference matmul matrix beginner 矩阵 入門 行列積' },
  'tutorials/spectrum': { title: words('Reconstruct a waveform', '从频谱重建波形', 'スペクトルから波形を復元'), desc: words('Take a signal into the frequency domain and back with a centered STFT.', '用居中短时傅里叶变换将信号转入频域，再重建波形。', '中心化 STFT で信号を周波数領域に移し、元の波形に戻します。'), terms:'audio STFT ISTFT FFT window Hann spectrum 频谱 逆变换 フーリエ 窓' },
  'tutorials/training': { title: words('Train a gain parameter', '训练增益参数', 'ゲインを学習する'), desc: words('Fit a gain to a target waveform with a reusable reverse-mode tape.', '用可复用的反向自动微分带拟合波形增益。', '再利用できる逆モード自動微分テープで波形のゲインを学習します。'), terms:'autograd tape SGD backward gradients optimization 自动微分 梯度 勾配 学習' },
  'docs/api': { title: words('Rust API', 'Rust API', 'Rust API'), desc: words('Explore working examples of the tensor, spectral and training APIs.', '通过示例了解张量、频谱和训练接口。', 'テンソル、スペクトル、学習の各 API をコード例で確認します。'), terms:'Tensor Shared Layout Tracked Builder SavedValue cuda Tape code reference nvrtc 示例 代码 リファレンス' },
  'docs/operators': { title: words('Operator coverage', '算子支持情况', '演算の対応状況'), desc: words('Look up forward and gradient support across CPU and NVIDIA CUDA.', '查询 CPU 与 NVIDIA CUDA 上的前向和梯度支持。', 'CPU と NVIDIA CUDA での順伝播と勾配計算の対応状況を調べます。'), terms:'coverage support STFT SDPA Bmm GRU Softmax Conv LayerNorm AdamW backward 支持 矩阵 対応' },
  'docs/architecture': { title: words('Architecture', '系统架构', 'アーキテクチャ'), desc: words('Follow a computation from checked Rust storage to native CPU or CUDA execution.', '了解一次计算如何从 Rust 张量存储走到 CPU 或 CUDA 执行。', 'Rust のストレージから CPU または CUDA で演算するまでの流れをたどります。'), terms:'layout storage execution fusion workspace graph 内存 融合 メモリ 実行' },
  'benchmarks': { title: words('Benchmark explorer', '性能基准浏览器', 'ベンチマークを比較'), desc: words('Explore recorded training and inference workloads alongside their PyTorch controls.', '按工作负载对照 fast-audio 与 PyTorch 的训练和推理实测记录。', '学習・推論の計測値をワークロードごとに PyTorch と比較できます。'), terms:'performance PyTorch NVIDIA L4 Intel CPU CUDA Graph GTCRN timings latency 性能 延迟 速度 レイテンシ' },
  'docs/roadmap': { title: words('Project roadmap', '项目路线图', 'ロードマップ'), desc: words('Track the single-device foundation and the work leading to larger audio models.', '了解单设备基础框架的进展，以及通往更大音频模型的后续工作。', '単一デバイスの基盤と、より大きな音声モデルに向けた今後の計画を紹介します。'), terms:'plan development phases BF16 future 计划 进展 計画' },
} as const;

export type Page = keyof typeof pages;
export type Tutorial = 'tutorials/tensors' | 'tutorials/spectrum' | 'tutorials/training';
export const tutorials: Tutorial[] = ['tutorials/tensors', 'tutorials/spectrum', 'tutorials/training'];
export const referencePages: Page[] = ['docs/api', 'docs/operators', 'docs/architecture', 'docs/roadmap'];
export const href = (lang: Lang, page: string = '') => '#/' + lang + (page ? '/' + page : '');

const aliases: Record<string, Page> = { overview: '', why: 'tutorials', quickstart: 'docs/install', architecture: 'docs/architecture', api: 'docs/api', coverage: 'docs/operators', roadmap: 'docs/roadmap', benchmarks: 'benchmarks' };
const isLang = (value: string | null): value is Lang => value === 'en' || value === 'zh' || value === 'ja';
const preferenceKey = 'fast-audio-language';

function preferredLang(): Lang {
  try {
    const saved = localStorage.getItem(preferenceKey);
    if (isLang(saved)) return saved;
  } catch { /* Browser storage is optional. */ }
  for (const tag of navigator.languages) {
    const lang = tag.toLowerCase().split('-')[0];
    if (isLang(lang)) return lang;
  }
  return 'en';
}

export function saveLanguage(lang: Lang) {
  try { localStorage.setItem(preferenceKey, lang); } catch { /* The URL retains this choice. */ }
}

export function readRoute() {
  const hash = location.hash.slice(1).replace(/^\/+|\/+$/g, '');
  const [first, ...segments] = hash.split('/');
  const lang = isLang(first) ? first : preferredLang();
  const path = isLang(first) ? segments.join('/') : (Object.hasOwn(aliases, hash) ? aliases[hash] : hash);
  const page = Object.hasOwn(pages, path) ? path as Page : null;
  return { lang, page, path };
}

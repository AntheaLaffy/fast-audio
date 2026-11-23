import { useState } from 'react';
import { Box, Cpu, GitBranch, Layers3, Workflow } from 'lucide-react';
import { words, type Lang } from '../data/i18n';

const layers = [
  { icon: Box, name: words('Tensor storage', '张量存储', 'テンソルの格納'), path: 'Tensor · Layout · Shared<T>', desc: words('Tensor owns contiguous host memory. A Layout describes signed strides and transformations independently of the payload. Shared owners allow checked reads and writes during training.', 'Tensor 拥有连续的主机内存。Layout 单独描述有符号步幅和变换。训练时，Shared 所有者提供经过检查的读写访问。', 'Tensor はホストの連続メモリを所有します。Layout はデータと独立に符号付きストライドや変換を記述し、Shared は学習時の読み書きを管理します。') },
  { icon: GitBranch, name: words('Computation graph', '计算图', '計算グラフ'), path: 'autograd::Builder · Program', desc: words('Leaves describe shapes and gradient requirements. The builder records operations and checks their geometry. Finishing the graph selects its root outputs and yields a reusable program.', '叶子节点描述形状和梯度需求。构建器记录算子并检查几何关系。选定根输出后，构建器返回可复用的程序。', 'リーフで形状と勾配の要否を指定します。ビルダーが演算と形状を検査し、根となる出力を選んで再利用可能なプログラムにします。') },
  { icon: Workflow, name: words('Training tape', '训练执行带', '学習テープ'), path: 'cpu::Tape · cuda::Tape', desc: words('Bind each leaf to its storage once. A prepared tape schedules forward operations and their local derivatives, reuses intermediate storage and updates the requested parameters.', '将每个叶子绑定到存储一次。预备好的 tape 调度前向运算与局部导数，复用中间存储并更新指定参数。', '各リーフにストレージを一度結び付けます。テープは順伝播と局所微分をスケジュールし、中間ストレージを再利用して指定のパラメーターを更新します。') },
  { icon: Layers3, name: words('Fusion and replay', '融合与重放', '融合とリプレイ'), path: 'autograd/fusion.rs · cuda/tape.rs', desc: words('Compatible pointwise, reduction and update operations share fused execution. CPU uses cache-sized tiles. CUDA emits kernels with NVRTC and can capture eligible training steps in a CUDA Graph.', '兼容的逐元素、归约与更新操作可融合执行。CPU 按缓存大小分块，CUDA 通过 NVRTC 生成内核，并能将符合条件的训练步骤捕获到 CUDA Graph。', '適合する要素演算・集約・更新を融合します。CPU はキャッシュ単位に分割し、CUDA は NVRTC でカーネルを生成して対象ステップを CUDA Graph に記録できます。') },
  { icon: Cpu, name: words('Backend execution', '后端执行', 'バックエンド実行'), path: 'Cpu::Reference · oneMKL · cuBLAS · cuFFT', desc: words('Prepare convolution and spectral plans for a specific geometry and backend. The caller supplies output tensors. A CUDA session owns its stream, with synchronization available at the host boundary.', '按后端和几何参数预备卷积与频谱计划。调用者提供输出张量。CUDA 会话拥有其计算流，可在主机边界显式同步。', '形状とバックエンドに合わせて畳み込みや FFT のプランを準備します。出力テンソルは呼び出し側が用意し、CUDA セッションのストリームはホスト境界で同期できます。') },
];
export function ArchitectureStack({ lang }: { lang: Lang }) {
  const [index, setIndex] = useState(0);
  const active = layers[index];
  return <div className="reading"><div className="layer-list">{layers.map(({icon:Icon,name},i) => <button key={name.en} aria-pressed={i === index} onClick={() => setIndex(i)}><Icon size={20}/>{name[lang]}</button>)}</div><article className="detail-box"><active.icon size={30}/><h2>{active.name[lang]}</h2><code>{active.path}</code><p>{active.desc[lang]}</p></article></div>;
}

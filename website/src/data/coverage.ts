import { words, type Words } from './i18n';

export type Status = 'verified' | 'inference' | 'planned';
type Operator = { name: string; cpu: string; gpu: string; status: Status; notes: Words };
export const operators: Operator[] = [
  {name:'Layout · CopyPlan', cpu:'Rust', gpu:'CUDA', status:'verified', notes:words('Signed strides, slices, broadcasts and permutations. Mutable output views must be injective.', '有符号步幅、切片、广播和维度置换。可变输出视图必须无重叠。', '符号付きストライド、スライス、ブロードキャスト、軸の置換。可変出力ではアドレスの重複を検査します。')},
  {name:'UnaryOp · BinaryOp', cpu:'Rust', gpu:'CUDA', status:'verified', notes:words('Pointwise transforms support local derivatives, including reduction of broadcast gradients.', '逐元素运算支持局部导数及广播梯度归约。', '要素演算は局所微分と、ブロードキャストした勾配の集約に対応します。')},
  {name:'Sum · Mean · Max · Min · LogSumExp', cpu:'Rust / oneMKL', gpu:'CUDA', status:'verified', notes:words('Explicit axes and keep-dimension policies. Extremum gradients account for ties.', '显式指定轴与维度保留策略，极值梯度处理并列值。', '集約軸と次元保持を明示します。極値の勾配は同値要素を考慮します。')},
  {name:'LayerNorm · RmsNorm · GroupNorm', cpu:'Rust / AVX2', gpu:'CUDA', status:'verified', notes:words('Prepared normalization with local gradients. Compose affine scale and bias separately.', '归一化计划支持局部梯度，仿射缩放和偏置可单独组合。', '正規化プランと局所勾配に対応します。アフィン変換のスケールとバイアスは個別に合成します。')},
  {name:'Softmax · LogSoftmax', cpu:'Rust / AVX2', gpu:'CUDA', status:'verified', notes:words('Stable max-shifted reductions along a selected axis support backward from the output.', '沿指定轴执行数值稳定的归约，可从输出计算反向。', '指定した軸で最大値を引いて安定化し、出力から逆伝播できます。')},
  {name:'Matmul · Bmm', cpu:'Rust / oneMKL', gpu:'cuBLAS / CUDA', status:'verified', notes:words('Matrix products and broadcast batch dimensions with gradients for both operands.', '矩阵乘与可广播的批量矩阵乘，两个操作数均支持梯度。', '行列積およびバッチ次元のブロードキャストと、両方の入力に対する勾配に対応します。')},
  {name:'Conv1d · Conv2d · ConvTranspose', cpu:'Rust / oneMKL / oneDNN', gpu:'cuBLAS / CUDA / cuDNN', status:'verified', notes:words('Grouped, strided and dilated convolutions with selective input and weight gradients.', '分组、步幅和空洞卷积支持按需计算输入与权重的梯度。', 'グループ、ストライド、膨張畳み込みで入力と重みの勾配を選択できます。')},
  {name:'Stft · CenteredStft · Istft', cpu:'RustFFT / oneMKL', gpu:'cuFFT / CUDA', status:'verified', notes:words('One-sided spectra use an explicit real/imaginary pair. Inverse plans check overlap coverage.', '单边频谱使用显式实部与虚部。逆变换计划会检查重叠覆盖。', '片側スペクトルの末尾次元は実部と虚部です。逆変換で窓の重なりを検査します。')},
  {name:'Pad · Cat', cpu:'Rust', gpu:'CUDA', status:'verified', notes:words('Constant, reflect and replicate padding. Concatenation supports selected-axis gradients.', '支持常量、反射和复制填充，拼接可沿所选轴回传梯度。', '定数・反射・複製パディングと、指定軸での連結の勾配に対応します。')},
  {name:'Sdpa · MHA · GQA', cpu:'Rust / AVX2', gpu:'CUDA', status:'verified', notes:words('Causal and non-causal attention with gradients for queries, keys and values.', '因果与非因果注意力均支持查询、键、值的梯度。', '因果・非因果の注意機構でクエリ、キー、値それぞれの勾配に対応します。')},
  {name:'Tape · SGD · AdamW', cpu:'Rust / oneMKL', gpu:'CUDA / NVRTC', status:'verified', notes:words('Reusable reverse-mode execution, explicit parameter updates, gradient clipping and checkpoints.', '可复用的反向模式执行、显式参数更新、梯度裁剪与检查点。', '再利用可能な逆モード実行、明示的なパラメーター更新、勾配クリッピング、チェックポイント。')},
  {name:'GRU', cpu:'Rust / oneMKL', gpu:'cuDNN', status:'inference', notes:words('Single-layer reset-after GRU with explicit state, as used in the GTCRN inference fixture.', '带显式状态的单层 reset-after GRU，用于 GTCRN 推理回归。', '明示的な状態を持つ単層 reset-after GRU。GTCRN の推論検証で使用します。')},
  {name:'Arbitrary attention masks', cpu:'—', gpu:'—', status:'planned', notes:words('General additive and boolean masks are a future extension to the verified causal and non-causal SDPA paths.', '通用加性掩码和布尔掩码是现有因果与非因果 SDPA 路径的后续扩展。', '一般の加算・ブールマスクは、検証済みの因果・非因果 SDPA を拡張する計画です。')},
];
export const statusNames = {
  verified: words('Verified gradients', '已验证梯度', '勾配検証済み'),
  inference: words('Inference only', '仅推理', '推論のみ'),
  planned: words('Planned', '规划中', '計画中'),
};

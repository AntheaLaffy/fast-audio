import { words } from './i18n';
import { type Tutorial } from './pages';
import tensorCode from '../examples/first_tensor.rs?raw';
import spectrumCode from '../examples/spectral_roundtrip.rs?raw';
import trainingCode from '../examples/train_gain.rs?raw';

export const lessons = {
  'tutorials/tensors': {
    name: 'first_tensor', code: tensorCode,
    idea: words('Each row holds three samples. The column of weights subtracts the last sample from the first. Allocate a two-row output, then pass it to the CPU backend.', '每一行包含三个采样点。权重列将首个采样值减去末个采样值。先分配两行输出，再交给 CPU 后端计算。', '各行には 3 つのサンプルがあります。重みの列は先頭の値から末尾の値を引きます。先に 2 行の出力を確保して CPU に渡します。'),
    check: words('The program prints [-2.0, -2.0]. Swap the first and last weights and update the assertion. Both outputs should become 2.0.', '程序会输出 [-2.0, -2.0]。交换第一个和最后一个权重，并更新断言，两个输出都应变为 2.0。', '出力は [-2.0, -2.0] です。最初と最後の重みを入れ替え、アサーションを更新すると、両方とも 2.0 になります。'),
  },
  'tutorials/spectrum': {
    name: 'spectral_roundtrip', code: spectrumCode,
    idea: words('Analyze 64 samples with a 16-point FFT and a four-sample hop. The centered analysis reflects the boundaries. Analysis and synthesis share a periodic Hann window and the same geometry.', '用 16 点 FFT 和 4 点帧移分析 64 个采样点。居中分析会反射填充边界，分析与合成共用周期 Hann 窗和相同的几何参数。', '64 サンプルを FFT 長 16、ホップ長 4 で解析します。中心化時は両端を反射して補い、解析と合成に同じ周期 Hann 窓と設定を使います。'),
    check: words('The maximum absolute reconstruction error should be below 1e-5. Try a hop of 8 and compare the error with the original run.', '最大绝对重建误差应小于 1e-5。把帧移改为 8，再与第一次的误差比较。', '最大絶対復元誤差が 1e-5 未満であることを確認します。ホップ長を 8 に変更して初回の誤差と比べてみましょう。'),
  },
  'tutorials/training': {
    name: 'train_gain', code: trainingCode,
    idea: words('The target is twice the input waveform. Declare a trainable scalar gain, measure mean squared error, and bind the tensors once. Each tape step runs the forward pass, backward pass and SGD update.', '目标波形是输入的两倍。声明一个可训练的标量增益，用均方误差作为损失，一次性绑定张量。每次执行 tape 都会完成前向、反向和 SGD 更新。', '目標波形は入力の 2 倍です。学習するスカラーのゲインと平均二乗誤差を定義し、テンソルを一度だけ結び付けます。各ステップで順伝播、逆伝播、SGD 更新を実行します。'),
    check: words('After 12 updates the gain is about 1.9995. Halve the learning rate and inspect how much closer the same 12 updates get to the target gain of 2.', '更新 12 次后，增益约为 1.9995。将学习率减半，比较相同更新次数后与目标增益 2 的距离。', '12 回更新するとゲインは約 1.9995 になります。学習率を半分にして、同じ更新回数で目標の 2 にどこまで近づくか確かめましょう。'),
  },
} satisfies Record<Tutorial, {name: string; code: string; idea: ReturnType<typeof words>; check: ReturnType<typeof words>}>;

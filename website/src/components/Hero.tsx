import { BookOpen, Box, CircuitBoard, Cpu, Layers3, Terminal } from 'lucide-react';
import { ui, words, type Lang } from '../data/i18n';
import { href } from '../data/pages';
import { AudioCanvas } from './AudioCanvas';

export function Hero({ lang }: { lang: Lang }) {
  return <section className="cover">
    <div className="frame cover-layout">
      <div className="cover-copy">
        <p className="cover-kicker"><span className="signal-rule" aria-hidden="true"/>{words('Rust audio framework', 'Rust 音频框架', 'Rust 音声フレームワーク')[lang]}</p>
        <h1>fast-audio</h1>
        <p className="cover-title">{words('Deep learning for audio.', '为音频而生的深度学习。', '音のための深層学習。')[lang]}</p>
        <p className="intro">{words('Build and train audio models in Rust. From raw waveforms to differentiable spectra, on Intel CPUs and NVIDIA GPUs.', '用 Rust 构建和训练音频模型。从原始波形到可微频谱，在 Intel CPU 和 NVIDIA GPU 上运行。', 'Rust で音声モデルを構築し、学習する。生の波形から微分可能なスペクトルまで、Intel CPU と NVIDIA GPU で扱えます。')[lang]}</p>
        <div className="cover-actions"><a className="btn-primary" href={href(lang, 'docs/install')}><Terminal size={18}/>{ui.start[lang]}</a><a className="btn-secondary" href={href(lang, 'tutorials')}><BookOpen size={17}/>{ui.browse[lang]}</a></div>
      </div>
      <AudioCanvas lang={lang}/>
    </div>
    <div className="spec-bar"><div className="frame spec-items">
      <span><Box/>Rust 2024</span><span><Cpu/>Intel oneMKL</span><span><CircuitBoard/>NVIDIA CUDA</span><span><Layers3/>{words('FP32 autograd', 'FP32 自动微分', 'FP32 自動微分')[lang]}</span>
    </div></div>
  </section>;
}

import { useState } from 'react';
import { BookOpen, Check, ChevronDown, Cpu, CircuitBoard, Microchip } from 'lucide-react';
import { href, pages } from '../data/pages';
import { ui, words, type Lang } from '../data/i18n';
import { CodeBlock } from './CodeBlock';
import { cratePage, previewVersion, repository } from '../data/release';

const platforms = [
  { id: 'portable', label: words('Portable CPU', '通用 CPU', '標準 CPU'), icon: Cpu },
  { id: 'intel', label: words('Intel CPU', 'Intel CPU', 'Intel CPU'), icon: Microchip },
  { id: 'nvidia', label: words('NVIDIA GPU', 'NVIDIA GPU', 'NVIDIA GPU'), icon: CircuitBoard },
] as const;
type Platform = typeof platforms[number]['id'];

export function QuickStart({ lang }: { lang: Lang }) {
  const [platform, setPlatform] = useState<Platform>('portable');
  const [source, setSource] = useState(false);
  const [oneDnn, setOneDnn] = useState(false);
  const [cuDnn, setCuDnn] = useState(false);
  const [arch, setArch] = useState('89');
  const feature = platform === 'intel' ? (oneDnn ? 'mkl,onednn' : 'mkl') : platform === 'nvidia' ? (cuDnn ? 'cudnn' : 'cuda') : '';
  const flags = feature ? ' --features ' + feature : '';
  const prefix = platform === 'nvidia' ? 'CUDA_ARCH=' + arch + ' ' : '';
  const build = prefix + 'cargo build --locked --release' + flags;
  const test = prefix + 'cargo test --locked' + flags;
  const requirements = {
    portable: words('Use the Rust toolchain and Cargo. The default backend uses RustFFT and runs on the CPU.', '安装 Rust 工具链和 Cargo。默认后端使用 RustFFT，在 CPU 上运行。', 'Rust ツールチェーンと Cargo を用意します。標準バックエンドは CPU 上で RustFFT を使います。'),
    intel: words('Install oneMKL headers and make libmkl_rt available to the dynamic loader. Set MKLROOT if oneMKL is installed outside its default location.', '安装 oneMKL 头文件，并让动态链接器能找到 libmkl_rt。安装在非默认路径时，请设置 MKLROOT。', 'oneMKL ヘッダーをインストールし、動的リンカーから libmkl_rt を参照できるようにします。標準以外に配置した場合は MKLROOT を設定します。'),
    nvidia: words('Install the NVIDIA driver, CUDA Toolkit, nvcc and a compatible C++ compiler. Set CUDA_HOST_CXX if you need to select a host compiler.', '安装 NVIDIA 驱动、CUDA Toolkit、nvcc 与兼容的 C++ 编译器。需要指定主机编译器时，设置 CUDA_HOST_CXX。', 'NVIDIA ドライバー、CUDA Toolkit、nvcc と対応する C++ コンパイラーを用意します。必要に応じて CUDA_HOST_CXX を設定します。'),
  };
  return <div className="reading">
    <div className="section-tabs"><button aria-pressed={!source} onClick={() => setSource(false)}>{words('Use the crate', '使用 crate', 'クレートを使う')[lang]}</button><button aria-pressed={source} onClick={() => setSource(true)}>{words('Build from source', '从源码构建', 'ソースからビルド')[lang]}</button></div>
    <div className="setup-panel"><h2>{words('Compute platform', '计算平台', '実行環境')[lang]}</h2>
    <div className="platforms">{platforms.map(({id, label, icon: Icon}) => <button key={id} aria-pressed={platform === id} onClick={() => setPlatform(id)}><Icon size={20}/>{label[lang]}{platform === id && <Check size={15}/>}</button>)}</div>
    {platform === 'intel' && <label className="addon"><input type="checkbox" checked={oneDnn} onChange={e => setOneDnn(e.target.checked)}/> {words('Enable oneDNN convolutions', '启用 oneDNN 卷积', 'oneDNN 畳み込みを有効にする')[lang]}</label>}
    {platform === 'nvidia' && <div className="gpu-options"><label className="addon"><input type="checkbox" checked={cuDnn} onChange={e => setCuDnn(e.target.checked)}/> {words('Enable cuDNN 9', '启用 cuDNN 9', 'cuDNN 9 を有効にする')[lang]}</label><label className="arch-select">{words('GPU target', 'GPU 架构', 'GPU ターゲット')[lang]}<span><select value={arch} onChange={e => setArch(e.target.value)}><option value="89">SM 8.9 · NVIDIA L4</option><option value="86">SM 8.6 · RTX 3060</option></select><ChevronDown size={14}/></span></label></div>}
    <h3>{ui.requirements[lang]}</h3><p>{requirements[platform][lang]} {platform === 'intel' && oneDnn && words('Also install the oneDNN development package.', '还需安装 oneDNN 开发包。', 'oneDNN 開発パッケージも必要です。')[lang]}{platform === 'nvidia' && cuDnn && words('Also install the cuDNN 9 headers and libraries.', '还需安装 cuDNN 9 的头文件及库。', 'cuDNN 9 のヘッダーとライブラリも必要です。')[lang]}</p></div>
    {source ? <><h2>{words('Build and verify', '构建与验证', 'ビルドと動作確認')[lang]}</h2>
    <p>{words('Run these commands in the fast-audio source directory containing Cargo.toml. The project uses Rust edition 2024 and its checked-in Cargo.lock.', '在含有 Cargo.toml 的 fast-audio 源码目录中运行以下命令。项目使用 Rust 2024 版和随源码提供的 Cargo.lock。', 'Cargo.toml がある fast-audio のソースディレクトリで実行します。Rust 2024 エディションと同梱の Cargo.lock を使います。')[lang]}</p>
    <CodeBlock lang={lang} shell code={'git clone --branch v' + previewVersion + ' ' + repository + '.git\ncd fast-audio\n' + build + '\n' + test}/>
    <h2>{words('Run an audio example', '运行音频示例', '音声のサンプルを実行')[lang]}</h2>
    <p>{words('The round-trip example writes input signals, spectra and reconstructed waveforms to a new directory. Choose an unused output path each time.', '往返重建示例会把输入信号、频谱和还原后的波形写入新目录。每次运行时选择一个尚不存在的输出目录。', '往復変換の例は入力、スペクトル、復元波形を新しいディレクトリに保存します。実行ごとに未使用の出力先を指定してください。')[lang]}</p>
    <CodeBlock lang={lang} shell code={prefix + 'cargo run --locked --release' + flags + ' --example audio-roundtrip -- ' + ({portable:'reference', intel:'mkl', nvidia:'cuda'}[platform]) + ' audio-output'}/>
    </> : <><h2>{words('Add the preview release', '添加预览版依赖', 'プレビュー版を追加')[lang]}</h2><p>{words('Run these commands in your Rust project. The exact version pins the preview. Rust 1.87 or later is required.', '在你的 Rust 项目中执行以下命令。精确版本号锁定此次预览版，需要 Rust 1.87 或更高版本。', 'Rust プロジェクト内で実行します。完全なバージョン指定でプレビュー版を固定します。Rust 1.87 以降が必要です。')[lang]}</p>
      <CodeBlock lang={lang} shell code={'cargo add fast-audio@=' + previewVersion + flags + '\n' + prefix + 'cargo check'}/><a className="text-link" href={cratePage}>crates.io · {previewVersion}</a></>}
    <div className="next-read"><small>{ui.next[lang]}</small><a className="text-link" href={href(lang,'tutorials/tensors')}><BookOpen size={17}/>{pages['tutorials/tensors'].title[lang]}</a></div>
  </div>;
}

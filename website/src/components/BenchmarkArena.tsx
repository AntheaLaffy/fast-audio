import { useState } from 'react';
import { Download } from 'lucide-react';
import { EVOLUTION_STAGES, GTCRN_CASES, INTEL_CPU_CASES, NVIDIA_L4_CASES, type WorkloadFamily } from '../data/benchmarks';
import { ui, words, type Lang } from '../data/i18n';

const tabs = { gpu: words('NVIDIA L4','NVIDIA L4','NVIDIA L4'), cpu: words('Intel CPU','Intel CPU','Intel CPU'), gtcrn: words('GTCRN inference','GTCRN 推理','GTCRN 推論'), history: words('History','优化历史','最適化の履歴') };
const families = {all:ui.all, dense:words('Dense','稠密','行列'), waveform:words('Waveform','波形','波形'), spectrum:words('Spectrum','频谱','スペクトル')};
const stageJa = [
  ['学習の初期ベースライン','CPU と GPU で共通の形状を固定し、勾配と SGD 更新を含む学習ステップを測定しました。'],
  ['CPU 逆伝播の準備','実行前に勾配カーネルを選択することで、対象の CPU ワークロードを約 1.56 倍高速化しました。'],
  ['集約とインデックス計算','小さい集約軸の CUDA カーネルと CPU の固定木集約を最適化しました。'],
  ['oneMKL VML と CUDA Graph','CPU に oneMKL のベクトル数学関数を導入し、CUDA では学習ステップを記録して再実行できるようにしました。'],
  ['学習ステップの融合','CPU のタイル処理と NVRTC 生成カーネルを導入しました。両者で順伝播、逆伝播、更新をまとめて実行します。'],
];

function Gtcrn({ lang }: { lang: Lang }) {
  const [mode,setMode] = useState<'cpu'|'eager'|'graph'>('graph');
  return <><p>{words('A full 16 kHz speech-enhancement model. Times are medians in milliseconds from the matched native and PyTorch runs at a7caf75.', '完整的 16 kHz 语音增强模型。表格列出 a7caf75 版本原生与 PyTorch 配对运行的中位数，单位为毫秒。', '16 kHz の音声強調モデル全体を対象に、a7caf75 で条件を揃えて計測した中央値をミリ秒で表示します。')[lang]}</p>
    <div className="choices">{(['cpu','eager','graph'] as const).map((key,i) => <button key={key} aria-pressed={key === mode} onClick={() => setMode(key)}>{['Intel CPU · Eager','NVIDIA · Eager','NVIDIA · CUDA Graph'][i]}</button>)}</div>
    <div className="table-scroll"><table><thead><tr><th>{words('Batch × samples','批量 × 采样数','バッチ × サンプル数')[lang]}</th><th>fast-audio (ms)</th><th>PyTorch (ms)</th></tr></thead><tbody>{GTCRN_CASES.map(row => {
      const [native, torch] = mode === 'cpu' ? [row.intelNativeMs,row.intelEagerMs] : mode === 'eager' ? [row.gpuNativeEagerCtxMs,row.gpuEagerMs] : [row.gpuNativeGraphCtxMs,row.gpuPreparedGraphMs];
      return <tr key={row.id}><th>{row.id.slice(1).replace('-',' × ')}</th><td>{native.toFixed(3)}</td><td>{torch.toFixed(3)}</td></tr>;
    })}</tbody></table></div></>;
}

export function BenchmarkArena({ lang }: { lang: Lang }) {
  const [tab,setTab] = useState<keyof typeof tabs>('gpu');
  const [family,setFamily] = useState<WorkloadFamily>('all');
  const [ratio,setRatio] = useState(false);
  const cpu = tab === 'cpu';
  const dataset = cpu ? INTEL_CPU_CASES : NVIDIA_L4_CASES;
  const rows = dataset.filter(row => family === 'all' || row.family === family);
  const csv = ['workload,fast_audio_us,pytorch_us,pytorch_over_fast_audio',...rows.map(row => [row.id,row.nativeUs,row.torchUs,(row.torchUs/row.nativeUs).toFixed(4)].join(','))].join('\n') + '\n';

  return <div className="reading benchmark"><div className="section-tabs">{(Object.keys(tabs) as (keyof typeof tabs)[]).map(key => <button key={key} aria-pressed={tab === key} onClick={() => setTab(key)}>{tabs[key][lang]}</button>)}</div>
    {tab === 'history' ? <div className="roadmap">{EVOLUTION_STAGES.map((stage,i) => <article key={stage.commit}><small>{stage.date} · {stage.commit}</small><h2>{lang === 'ja' ? stageJa[i][0] : (lang === 'zh' ? stage.titleZh : stage.titleEn).split(' · ').pop()}</h2><p>{lang === 'ja' ? stageJa[i][1] : lang === 'zh' ? stage.summaryZh : stage.summaryEn}</p></article>)}</div> :
    tab === 'gtcrn' ? <Gtcrn lang={lang}/> : <>
    <div className="bench-snapshot"><div><span className="eyebrow">training-v1 · 2026-11-20</span><h2>{cpu ? 'Intel oneMKL' : 'NVIDIA L4'}</h2><p>{cpu ? words('Single-threaded oneMKL against PyTorch Inductor.', '单线程 oneMKL 对比 PyTorch Inductor。', 'シングルスレッドの oneMKL と PyTorch Inductor の比較。')[lang] : words('Captured training steps against PyTorch compiled CUDA Graph.', '捕获完整训练步骤，与 PyTorch 编译后的 CUDA Graph 对照。', '記録済みの学習ステップと PyTorch のコンパイル済み CUDA Graph を比較します。')[lang]}</p></div><div className="bench-stat"><strong>{cpu ? '4.22' : '1.61'}<i>×</i></strong><span>{words('Geometric mean speedup','几何平均加速比','幾何平均の高速化率')[lang]}</span></div></div>
    <p>{words('Five paired blocks, strict FP32, PyTorch 2.14.1 and fast-audio e6ad598. One timed step includes forward, backward and SGD. The aggregate covers all 12 workloads.', '五组配对测量，严格 FP32，使用 PyTorch 2.14.1 和 fast-audio e6ad598。每步计入前向、反向和 SGD 更新。汇总值涵盖全部 12 项负载。', '厳密な FP32 で PyTorch 2.14.1 と fast-audio e6ad598 を 5 ブロックずつ比較しました。各ステップに順伝播、逆伝播、SGD 更新を含み、集計値は 12 項目全体のものです。')[lang]}</p>
    <div className="bench-controls"><div className="choices">{(Object.keys(families) as WorkloadFamily[]).map(key => <button key={key} aria-pressed={family === key} onClick={() => setFamily(key)}>{families[key][lang]}</button>)}</div><div className="choices metric">{[false,true].map(value => <button key={String(value)} aria-pressed={value === ratio} onClick={() => setRatio(value)}>{(value ? words('Speedup','加速比','速度比') : words('Latency','耗时','レイテンシ'))[lang]}</button>)}</div></div>
    <div className="table-scroll"><table className="bench-table"><thead><tr><th>{words('Workload','工作负载','ワークロード')[lang]}</th>{ratio ? <th>PyTorch / fast-audio</th> : <><th>fast-audio (µs)</th><th>PyTorch (µs)</th></>}</tr></thead><tbody>{rows.map(row => <tr key={row.id}><th>{row.id}</th>{ratio ? <td><b className={row.torchUs < row.nativeUs ? 'under' : ''}>{(row.torchUs / row.nativeUs).toFixed(2)}×</b></td> : [row.nativeUs,row.torchUs].map((n,i) => <td key={i}><span>{n.toFixed(2)}</span><i aria-hidden="true" className={i ? 'torch-bar' : 'native-bar'} style={{width: n/Math.max(row.nativeUs,row.torchUs)*100+'%'}}/></td>)}</tr>)}</tbody></table></div>
    <p className="chart-note">{(ratio ? words('Above 1× means lower fast-audio latency.','高于 1× 表示 fast-audio 耗时更低。','1 倍を超えると fast-audio の実行時間が短くなります。') : words('Shorter bars mean lower latency. Each pair shares its own scale.', '条形越短表示耗时越低。每行的两条数据共用一个刻度。','各行の棒は共通の尺度で、短いほど低レイテンシです。'))[lang]}</p>
    <a className="text-link" download={'training-v1-'+tab+'-'+family+'.csv'} href={'data:text/csv;charset=utf-8,'+encodeURIComponent(csv)}><Download size={16}/>{words('Download these results','下载当前结果','表示中の結果をダウンロード')[lang]}</a>
    </>}
  </div>;
}

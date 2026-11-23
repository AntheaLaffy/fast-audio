import { useEffect, useRef } from 'react';
import { BookOpen, ChartNoAxesColumnIncreasing, Code2, Workflow } from 'lucide-react';
import { href, pages, tutorials, type Page } from '../data/pages';
import { ui, words, type Lang } from '../data/i18n';
import { NVIDIA_L4_CASES } from '../data/benchmarks';

type Cover = 'tensor' | 'spectrum' | 'graph';
const covers: Cover[] = ['tensor', 'spectrum', 'graph'];

function drawCover(ctx: CanvasRenderingContext2D, type: Cover) {
  ctx.clearRect(0, 0, 360, 125);
  if (type === 'tensor') {
    const pt = (x: number, y: number, z: number) => [180 + (x - y) * 14, 84 + (x + y - 6) * 6.5 - z * 10];
    for (let z = 0; z < 6; z++) {
      ctx.fillStyle = z === 5 ? '#e99667' : '#eaa47b';
      ctx.strokeStyle = '#713b29'; ctx.lineWidth = 1;
      ctx.beginPath();
      [[0,0], [6,0], [6,6], [0,6]].forEach(([x,y], index) => { const p = pt(x,y,z); if (!index) ctx.moveTo(p[0],p[1]); else ctx.lineTo(p[0],p[1]); });
      ctx.closePath(); ctx.fill(); ctx.stroke();
      for (let i = 1; i < 6; i++) {
        for (const [a,b] of [[pt(i,0,z),pt(i,6,z)], [pt(0,i,z),pt(6,i,z)]]) {
          ctx.beginPath(); ctx.moveTo(a[0],a[1]); ctx.lineTo(b[0],b[1]); ctx.stroke();
        }
      }
    }
  } else if (type === 'spectrum') {
    for (let i = 0; i < 110; i++) {
      const x = i / 109, wave = Math.sin(Math.PI*x) ** 1.4;
      const h = (6 + 46 * wave * (0.25 + 0.75 * Math.abs(Math.sin(x*16)*Math.cos(x*7.6))) + 6*Math.sin(i*1.73)**2) * Math.sin(Math.PI*x);
      const px = 20 + x*320;
      ctx.strokeStyle = '#326756'; ctx.lineWidth = 1.55;
      ctx.beginPath(); ctx.moveTo(px, 63-h); ctx.lineTo(px,63+h); ctx.stroke();
    }
  } else {
    ctx.lineWidth = 1.1; ctx.strokeStyle = '#6d5782';
    const layers = [[63, 2], [132, 4], [209, 3], [290, 2]];
    const points = layers.map(([x,n]) => Array.from({length:n},(_,i) => [x, 62 + (i-(n-1)/2)*25]));
    for (let i=1;i<points.length;i++) for (const a of points[i-1]) for (const b of points[i]) {
      ctx.beginPath(); ctx.moveTo(a[0],a[1]); ctx.bezierCurveTo(a[0]+30,a[1],b[0]-30,b[1],b[0],b[1]); ctx.stroke();
    }
    for (const [index, layer] of points.entries()) for (const [x,y] of layer) {
      ctx.beginPath(); ctx.arc(x,y,index===3?5:3.7,0,2*Math.PI); ctx.fillStyle = index===3?'#594369':'#d0c2dc'; ctx.fill(); ctx.stroke();
    }
  }
}

function GuideArt({ type }: { type: Cover }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const el = ref.current, ctx = el?.getContext('2d');
    if (!el || !ctx) return;
    el.width = 720; el.height = 250; ctx.setTransform(2,0,0,2,0,0); drawCover(ctx, type);
  }, [type]);
  return <div className={'guide-art art-' + type} aria-hidden="true"><canvas ref={ref}/></div>;
}

export function GuideCards({ lang }: { lang: Lang }) {
  const categories = [words('Tensors', '张量', 'テンソル'), words('Signal processing', '信号处理', '信号処理'), words('Autograd', '自动微分', '自動微分')];
  return <div className="guide-cards">{tutorials.map((page, i) => <a key={page} href={href(lang, page)}>
    <GuideArt type={covers[i]}/><div className="guide-body"><span className="eyebrow">{categories[i][lang]}</span><h3>{pages[page].title[lang]}</h3><p>{pages[page].desc[lang]}</p></div>
  </a>)}</div>;
}

export function StoryPillars({ lang }: { lang: Lang }) {
  const sample = NVIDIA_L4_CASES.find(row => row.id === 'spectrum-large')!;
  const links: { page: Page; icon: typeof BookOpen }[] = [{page:'docs/api',icon:Code2},{page:'docs/operators',icon:BookOpen},{page:'docs/architecture',icon:Workflow}];
  return <>
    <section className="field-guide frame"><div className="section-head"><h2>{words('Developer guides', '开发者指南', '開発者ガイド')[lang]}</h2><a className="text-link" href={href(lang, 'tutorials')}><BookOpen size={16}/>{ui.browse[lang]}</a></div><GuideCards lang={lang}/></section>
    <section className="performance-section"><div className="frame performance-layout">
      <div><p className="eyebrow">{ui.benchmarks[lang]}</p><h2>{words('Inside the training step', '看清每一步训练', '学習ステップを測る')[lang]}</h2>
      <p>{words('Compare complete forward, backward and SGD timings across dense, waveform and spectrum workloads.', '按稠密、波形和频谱三类负载，对照一次完整前向、反向和 SGD 更新的耗时。', '行列・波形・スペクトルの各ワークロードで、順伝播から逆伝播、SGD 更新までの時間を比較できます。')[lang]}</p>
      <a className="text-link" href={href(lang,'benchmarks')}><ChartNoAxesColumnIncreasing size={17}/>{pages.benchmarks.title[lang]}</a></div>
      <figure className="timing-card"><figcaption><strong>{sample.id}</strong><span>NVIDIA L4 · FP32</span></figcaption>
        {[[sample.nativeUs, 'fast-audio'], [sample.torchUs, 'PyTorch CUDA Graph']].map(([n, name], i) => <div className="timing-row" key={name}><div><span>{name}</span><b>{Number(n).toFixed(2)} µs</b></div><div className="timing-track"><i className={i ? 'comparison' : ''} style={{width: Number(n)/sample.torchUs*100+'%'}}/></div></div>)}
        <small>training-v1 · e6ad598 · 2026-11-20</small>
      </figure>
    </div></section>
    <section className="frame reference-links"><h2>{ui.reference[lang]}</h2><div>{links.map(({page,icon:Icon}) => <a key={page} href={href(lang,page)}><Icon size={21}/>{pages[page].title[lang]}</a>)}</div></section>
  </>;
}

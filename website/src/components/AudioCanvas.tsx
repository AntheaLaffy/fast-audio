import { useEffect, useRef, useState } from 'react';
import { AudioWaveform, Pause, Play } from 'lucide-react';
import { Matrix4, Vector3 } from 'three';
import { words, type Lang } from '../data/i18n';

const presets = [
  words('Harmonic', '谐波', '倍音'),
  words('Formant', '共振峰', 'フォルマント'),
  words('Transient', '瞬态', '過渡音'),
];

const peak = (x: number, center: number, width: number) => Math.exp(-((x - center) ** 2) / (width ** 2));

function drawSurface(ctx: CanvasRenderingContext2D, width: number, height: number, preset: number, phase: number) {
  ctx.clearRect(0, 0, width, height);
  const rows = 83, columns = 172;
  // Axonometric projection of the frequency, amplitude and time coordinates.
  const projection = new Matrix4().set(
    width * 0.76, 0, -width * 0.22, width * 0.24,
    -height * 0.24, -height * 0.68, height * 0.29, height * 0.68,
    0, 0, 1, 0, 0, 0, 0, 1,
  );
  const point = new Vector3();
  const project = (x: number, z: number, amplitude: number) => point.set(x, amplitude, z).applyMatrix4(projection);
  // The illustration samples a different spectral envelope for each preset.
  for (let row = 0; row < rows; row++) {
    const z = row / (rows - 1);
    const depth = Math.sin(Math.PI * z) ** 0.85;
    const drift = Math.sin(z * 5.4 + phase) * 0.09;
    const a = 0.36 + drift + 0.18 * z;
    const b = 0.68 + Math.cos(3.8 * z - phase * 0.7) * 0.07;
    const ridge = new Path2D();
    for (let column = 0; column <= columns; column++) {
      const x = column / columns;
      const tex = 0.036 * Math.sin(x * 107 + Math.sin(z * 9 + phase) * 2.6)
        + 0.025 * Math.sin(x * 178 - z * 13 + phase);
      let signal: number;
      if (preset === 1) {
        signal = 0.71 * peak(x, 0.24 + drift, 0.052) + 0.9 * peak(x, a + 0.04, 0.070) + 0.65 * peak(x, b + 0.06, 0.055);
      } else if (preset === 2) {
        signal = (0.85 * peak(z, 0.43, 0.15) + 0.24 * peak(z, 0.72, 0.09)) * Math.exp(-Math.max(0, x - 0.2) * 2.2)
          * (0.79 + 0.23 * Math.cos(x * 48 - z * 12)) * Math.min(1, x * 7);
      } else {
        signal = 0.98 * peak(x, a, 0.125) + 0.60 * peak(x, b, 0.09) + 0.25 * peak(x, 0.20 + drift * 0.6, 0.075);
      }
      const amp = depth * Math.sin(Math.PI * x) * signal * (0.94 + tex + 0.10 * Math.sin(x * 63 - z * 7 + phase));
      const {x: px, y: py} = project(x, z, amp);
      if (column === 0) ridge.moveTo(px, py); else ridge.lineTo(px, py);
    }
    const {x: lx, y: ly} = project(0, z, 0);
    const {x: rx, y: ry} = project(1, z, 0);
    const mask = new Path2D(ridge);
    mask.lineTo(rx, ry + height * 0.015);
    mask.lineTo(lx, ly + height * 0.015);
    mask.closePath();
    ctx.fillStyle = '#191918';
    ctx.fill(mask);
    const ink = ctx.createLinearGradient(0, height * 0.98, width * 0.39, height * 0.04);
    ink.addColorStop(0, '#6d4937');
    ink.addColorStop(0.23, '#bd5c38');
    ink.addColorStop(0.49, '#ff7944');
    ink.addColorStop(0.73, '#ffab74');
    ink.addColorStop(1, '#ffe3b9');
    ctx.strokeStyle = ink;
    ctx.lineWidth = width < 430 ? 0.9 : 1.06;
    ctx.stroke(ridge);
  }
}

export function AudioCanvas({ lang }: { lang: Lang }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const phase = useRef(0.3);
  const [preset, setPreset] = useState(0);
  const [playing, setPlaying] = useState(() => !matchMedia('(prefers-reduced-motion: reduce)').matches);
  useEffect(() => {
    const el = canvas.current;
    const ctx = el?.getContext('2d');
    if (!el || !ctx) return;
    let frame = 0, previous = 0;
    let visible = false;
    function draw(now: number) {
      if (!el || !ctx) return;
      const w = el.clientWidth, h = el.clientHeight, scale = Math.min(devicePixelRatio, 2);
      if (el.width !== Math.round(w * scale) || el.height !== Math.round(h * scale)) {
        el.width = Math.round(w * scale); el.height = Math.round(h * scale);
      }
      ctx.setTransform(scale, 0, 0, scale, 0, 0);
      if (playing && previous) phase.current += Math.min(now - previous, 100) * 0.00020;
      previous = now;
      drawSurface(ctx, w, h, preset, phase.current);
    }
    function tick(now: number) {
      if (now - previous >= 62) draw(now);
      frame = requestAnimationFrame(tick);
    }
    function update() {
      cancelAnimationFrame(frame);
      previous = 0;
      draw(performance.now());
      if (visible && playing && !document.hidden) frame = requestAnimationFrame(tick);
    }
    const observer = new IntersectionObserver(([entry]) => { visible = entry.isIntersecting; update(); });
    const size = new ResizeObserver(update);
    observer.observe(el); size.observe(el);
    document.addEventListener('visibilitychange', update);
    return () => { cancelAnimationFrame(frame); observer.disconnect(); size.disconnect(); document.removeEventListener('visibilitychange', update); };
  }, [preset, playing]);

  return <figure className="monitor">
    <figcaption><span><AudioWaveform size={17}/>{words('Signal explorer', '信号浏览器', '信号ビューア')[lang]}</span><small>{words('Synthetic spectrum', '合成频谱', '合成スペクトル')[lang]}</small></figcaption>
    <div className="plot">
      <canvas ref={canvas} role="img" aria-label={presets[preset][lang] + ' · ' + words('frequency over time', '频率随时间变化', '周波数の時間変化')[lang]}/>
      <i className="corner tl"/><i className="corner tr"/><i className="corner bl"/><i className="corner br"/>
    </div>
    <div className="monitor-controls"><div className="presets">{presets.map((item, index) => <button key={item.en} aria-pressed={preset === index} onClick={() => setPreset(index)}>{item[lang]}</button>)}</div>
      <button className="play" aria-label={(playing ? words('Pause animation', '暂停动画', 'アニメーションを一時停止') : words('Play animation', '播放动画', 'アニメーションを再生'))[lang]} onClick={() => setPlaying(!playing)}>{playing ? <Pause size={15}/> : <Play size={15}/>}</button></div>
  </figure>;
}

import { useEffect, useRef, useState } from 'react';
import { Check, Copy, Download, FileCode2, Terminal } from 'lucide-react';
import { SyntaxHighlight } from './SyntaxHighlight';
import { ui, words, type Lang } from '../data/i18n';

export function CodeBlock({ lang, code, filename, shell = false, download = false }: { lang: Lang; code: string; filename?: string; shell?: boolean; download?: boolean }) {
  const [status, setStatus] = useState<'idle'|'done'|'error'>('idle');
  const timer = useRef<number | undefined>(undefined);
  useEffect(() => { setStatus('idle'); return () => clearTimeout(timer.current); }, [code]);
  async function copy() {
    clearTimeout(timer.current);
    try {
      await navigator.clipboard.writeText(code.trimEnd());
      setStatus('done');
    } catch { setStatus('error'); }
    timer.current = window.setTimeout(() => setStatus('idle'), 3000);
  }
  return <div className="code-block"><div className="code-top">
    <span>{shell ? <Terminal size={16}/> : <FileCode2 size={16}/>} {filename ?? 'Shell'}</span>
    <div>{download && <a aria-label={words('Download example', '下载示例', 'コード例をダウンロード')[lang]} href={'data:text/plain;charset=utf-8,' + encodeURIComponent(code.trimEnd() + '\n')} download={filename}><Download size={16}/></a>}
      <button onClick={copy}>{status === 'done' ? <Check size={15}/> : <Copy size={15}/>}<span aria-live="polite">{(status === 'done' ? ui.copied : ui.copy)[lang]}</span></button></div>
    </div><SyntaxHighlight code={code.trimEnd()} language={shell ? 'shell' : 'rust'}/>
    {status === 'error' && <p className="copy-message" role="status">{ui.copyError[lang]}</p>}
  </div>;
}

import { useState } from 'react';
import { ChevronDown, Search } from 'lucide-react';
import index from '../data/api-index.json';
import { CODE_EXAMPLES } from '../data/codeExamples';
import { ui, words, type Lang } from '../data/i18n';
import { CodeBlock } from './CodeBlock';

const examplesJa: Record<string, string> = { 'first-tensor': '最初のテンソル', 'fusion-tape': '学習テープ', 'cuda-graph': 'CUDA Graph', 'spectral-diff': '微分可能な STFT', 'views-versions': 'ビューと世代管理' };
const primary = ['Tensor', 'Layout', 'Cpu', 'Builder', 'Tape', 'CenteredStft', 'Istft', 'Shared'];
const entries = [...index].sort((a,b) => {
  const rank = (name: string) => primary.includes(name) ? primary.indexOf(name) : primary.length;
  return rank(a.name) - rank(b.name) || a.name.localeCompare(b.name);
});
const fullName = (item: typeof entries[number]) => [item.module,item.name].filter(Boolean).join('::');

export function CodeShowcase({ lang, initialQuery }: { lang: Lang; initialQuery: string }) {
  const [query, setQuery] = useState(initialQuery);
  const [limit, setLimit] = useState(8);
  const [exampleId, setExampleId] = useState(CODE_EXAMPLES[0].id);
  const example = CODE_EXAMPLES.find(e => e.id === exampleId)!;
  const term = query.trim().toLowerCase();
  const results = entries.filter(item => fullName(item).toLowerCase().includes(term) || item.methods.some(method => method.toLowerCase().includes(term)));
  const label = words('Find a type or method', '查找类型或方法', '型・メソッドを検索')[lang];
  return <div className="reading"><h2>{words('Public API index', '公共 API 索引', '公開 API インデックス')[lang]}</h2>
    <p>{words('Signatures come from Rustdoc for the default CPU build. Expand a symbol to see its declaration and methods.', '签名取自默认 CPU 构建的 Rustdoc。展开符号可查看声明及方法。', '標準 CPU ビルドの Rustdoc から抽出したシグネチャです。項目を開くと宣言とメソッドを確認できます。')[lang]}</p>
    <label className="filter-input"><Search size={18}/><input type="search" value={query} aria-label={label} placeholder={label} onChange={e => { setQuery(e.target.value); setLimit(8); }}/></label>
    <div className="api-index">{results.slice(0,limit).map(item => <details key={fullName(item)}><summary><code>{fullName(item)}</code><ChevronDown size={16}/></summary><div className="signatures"><code>{item.decl}</code>{item.methods.map((method,i) => <code key={i}>{method}</code>)}</div></details>)}</div>
    {!results.length && <p className="empty">{ui.noResults[lang]}</p>}
    {results.length > limit && <button className="quiet-button" onClick={() => setLimit(limit + 20)}>{words('Show more symbols', '显示更多符号', 'さらに表示')[lang]} <span>({limit}/{results.length})</span></button>}
    <h2>{words('API examples', 'API 示例', 'API の使用例')[lang]}</h2>
    <div className="choices">{CODE_EXAMPLES.map(ex => <button key={ex.id} aria-pressed={example.id === ex.id} onClick={() => setExampleId(ex.id)}>{({en:ex.tabEn, zh:ex.tabZh, ja:examplesJa[ex.id]})[lang]}</button>)}</div>
    <CodeBlock code={example.code} lang={lang} filename={example.filename}/>
    <p>{words('Generate the complete reference for the features enabled in your checkout with Cargo.', '使用 Cargo 可为当前源码中启用的功能生成完整文档。', '手元のソースで有効にした機能の詳細なリファレンスは Cargo で生成できます。')[lang]}</p><CodeBlock lang={lang} shell code="cargo doc --locked --no-deps --open"/>
  </div>;
}

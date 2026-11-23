import { useState } from 'react';
import { Search } from 'lucide-react';
import { operators, statusNames, type Status } from '../data/coverage';
import { ui, words, type Lang } from '../data/i18n';

export function CoverageMatrix({ lang }: { lang: Lang }) {
  const [filter, setFilter] = useState<Status|'all'>('all');
  const [query, setQuery] = useState('');
  const label = words('Find an operator or backend', '搜索算子或后端', '演算またはバックエンドを検索')[lang];
  const matched = operators.filter(row => (filter === 'all' || row.status === filter) && [row.name,row.cpu,row.gpu,...Object.values(row.notes)].join(' ').toLowerCase().includes(query.trim().toLowerCase()));
  return <div className="reading"><p>{words('The table tracks the strict FP32 foundation. A verified gradient has an explicit local derivative on CPU and CUDA.', '下表记录严格 FP32 基础算子。已验证梯度表示 CPU 和 CUDA 均有经过验证的显式局部导数。', '表には厳密な FP32 基盤の対応状況を示します。勾配検証済みの演算には、CPU と CUDA 双方の明示的な局所微分があります。')[lang]}</p>
    <div className="choices">{(['all','verified','inference','planned'] as const).map(key => <button aria-pressed={filter === key} key={key} onClick={() => setFilter(key)}>{(key === 'all' ? ui.all : statusNames[key])[lang]}</button>)}</div>
    <label className="filter-input"><Search size={18}/><input type="search" value={query} onChange={e => setQuery(e.target.value)} placeholder={label} aria-label={label}/></label>
    <div className="operator-list">{matched.map(row => <article key={row.name}><div><h2>{row.name === 'Arbitrary attention masks' ? words(row.name,'任意注意力掩码','任意の注意マスク')[lang] : row.name}</h2><span className={'status status-' + row.status}>{statusNames[row.status][lang]}</span></div><p>{row.notes[lang]}</p><small>CPU <b>{row.cpu}</b><span>GPU <b>{row.gpu}</b></span></small></article>)}
    {!matched.length && <p className="empty">{ui.noResults[lang]}</p>}</div>
  </div>;
}

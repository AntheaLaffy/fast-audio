import { Brand } from './Navbar';
import { href, pages, type Page } from '../data/pages';
import { ui, words, type Lang } from '../data/i18n';
import { cratePage, repository } from '../data/release';

export function Footer({ lang }: { lang: Lang }) {
  const groups: { title: string; links: Page[] }[] = [
    { title: ui.learn[lang], links: ['docs/install', 'tutorials', 'benchmarks'] },
    { title: ui.reference[lang], links: ['docs/api', 'docs/operators', 'docs/architecture', 'docs/roadmap'] },
  ];
  return <footer className="site-footer"><div className="frame footer-inner">
    <div><Brand lang={lang}/><p>{words('A Rust framework for audio training and inference.', '用于音频训练和推理的 Rust 框架。', '音声の学習と推論のための Rust フレームワーク。')[lang]}</p><div className="external-links"><a href={repository}>GitHub</a><a href={cratePage}>crates.io</a><a href={repository + '/blob/main/LICENSE'}>MIT</a></div></div>
    {groups.map(group => <nav key={group.title} aria-label={group.title}><h2>{group.title}</h2>{group.links.map(page => <a href={href(lang, page)} key={page}>{pages[page].title[lang]}</a>)}</nav>)}
  </div></footer>;
}

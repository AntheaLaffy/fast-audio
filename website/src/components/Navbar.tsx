import { useEffect, useRef, useState } from 'react';
import { AudioLines, BookOpen, ChevronDown, FileText, Globe2, Menu, Search, X } from 'lucide-react';
import { languages, localeTag, ui, words, type Lang } from '../data/i18n';
import { href, pages, saveLanguage, type Page } from '../data/pages';

export function Brand({ lang }: { lang: Lang }) {
  return <a href={href(lang)} className="wordmark" aria-label={'fast-audio · ' + ui.home[lang]}>
    <AudioLines size={30} strokeWidth={3} aria-hidden="true" /><span>fast-audio</span>
  </a>;
}

export function Navbar({ lang, path }: { lang: Lang; path: string }) {
  const [menu, setMenu] = useState(false);
  const [query, setQuery] = useState('');
  const dialog = useRef<HTMLDialogElement>(null);
  const focusSearch = useRef<HTMLInputElement>(null);
  const trigger = useRef<HTMLButtonElement>(null);
  function openSearch() {
    setMenu(false);
    setQuery('');
    if (!dialog.current?.open) dialog.current?.showModal();
    focusSearch.current?.focus();
  }
  useEffect(() => {
    const shortcut = (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k') {
        event.preventDefault();
        if (dialog.current?.open) dialog.current.close();
        else openSearch();
      }
      if (event.key === 'Escape') setMenu(false);
    };
    window.addEventListener('keydown', shortcut);
    return () => window.removeEventListener('keydown', shortcut);
  }, []);
  useEffect(() => { setMenu(false); dialog.current?.close(); }, [path, lang]);

  const links = [
    { page: 'docs/install', label: ui.docs, active: path.startsWith('docs/') },
    { page: 'tutorials', label: ui.tutorials, active: path.startsWith('tutorials') },
    { page: 'benchmarks', label: ui.benchmarks, active: path === 'benchmarks' },
  ];
  const tokens = query.toLocaleLowerCase().trim().split(/\s+/).filter(Boolean);
  const found = (Object.keys(pages) as Page[]).filter(page => {
    const entry = pages[page];
    const words = [...Object.values(entry.title), ...Object.values(entry.desc), entry.terms].join(' ').toLocaleLowerCase();
    return page && tokens.every(token => words.includes(token));
  });

  return <header className="site-nav">
    <a href="#main-content" className="skip-link" onClick={e => { e.preventDefault(); document.getElementById('main-content')?.focus(); }}>{ui.skip[lang]}</a>
    <div className="frame nav-inner">
      <Brand lang={lang} />
      <nav className="nav-links" aria-label={ui.menu[lang]}>{links.map(link =>
        <a href={href(lang, link.page)} key={link.page} aria-current={link.active ? 'location' : undefined}>{link.label[lang]}</a>)}</nav>
      <div className="nav-tools">
        <button ref={trigger} className="search-trigger" onClick={openSearch} aria-label={ui.searchDocs[lang]} title={ui.searchDocs[lang]}><Search size={18}/></button>
        <label className="language"><Globe2 size={16}/><select aria-label={ui.language[lang]} value={lang} onChange={e => {
          const value = e.target.value as Lang;
          saveLanguage(value);
          location.hash = href(value, path);
        }}>{(Object.keys(languages) as Lang[]).map(value => <option key={value} lang={localeTag[value]} value={value}>{languages[value]}</option>)}</select><ChevronDown size={13}/></label>
        <a className="nav-start" href={href(lang, 'docs/install')}>{ui.start[lang]}</a>
        <button className="mobile-menu" aria-expanded={menu} aria-controls="mobile-nav" aria-label={ui.menu[lang]} onClick={() => setMenu(!menu)}>{menu ? <X/> : <Menu/>}</button>
      </div>
    </div>
    {menu && <nav id="mobile-nav" aria-label={ui.menu[lang]}>{links.map(link => <a key={link.page} onClick={() => setMenu(false)} href={href(lang, link.page)}>{link.label[lang]}</a>)}</nav>}
    <dialog ref={dialog} className="search-dialog" aria-label={ui.searchDocs[lang]} onClose={() => trigger.current?.focus({ preventScroll: true })} onClick={e => { if (e.target === e.currentTarget) dialog.current?.close(); }}>
      <div className="search-sheet">
        <div className="search-entry"><Search/><input ref={focusSearch} aria-label={ui.searchDocs[lang]} placeholder={ui.searchDocs[lang]} value={query} onChange={e => setQuery(e.target.value)} onKeyDown={e => {
          if (e.key === 'Enter' && found.length) { location.hash = href(lang, found[0]); dialog.current?.close(); }
          if (e.key === 'ArrowDown') { e.preventDefault(); dialog.current?.querySelector<HTMLAnchorElement>('.search-results a')?.focus(); }
        }}/><button onClick={() => dialog.current?.close()} aria-label={ui.close[lang]}><X size={20}/></button></div>
        <div className="search-results"><p className="eyebrow">{(query ? words('Results', '搜索结果', '検索結果') : words('Explore', '浏览', 'ページ一覧'))[lang]}</p>
          {found.map(page => <a href={href(lang, page)} key={page} onClick={() => dialog.current?.close()}>{page.startsWith('tutorials') ? <BookOpen size={18}/> : <FileText size={18}/>}<span><strong>{pages[page].title[lang]}</strong><small>{pages[page].desc[lang]}</small></span></a>)}
          {!found.length && <p className="empty">{ui.noResults[lang]}</p>}
        </div>
      </div>
    </dialog>
  </header>;
}

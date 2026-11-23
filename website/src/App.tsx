import { useEffect, useRef, useState } from 'react';
import { ChevronDown, ChevronRight, Menu } from 'lucide-react';
import { Navbar } from './components/Navbar';
import { Hero } from './components/Hero';
import { StoryPillars, GuideCards } from './components/StoryPillars';
import { BenchmarkArena } from './components/BenchmarkArena';
import { ArchitectureStack } from './components/ArchitectureStack';
import { CodeShowcase } from './components/CodeShowcase';
import { CoverageMatrix } from './components/CoverageMatrix';
import { RoadmapSection } from './components/RoadmapSection';
import { QuickStart } from './components/QuickStart';
import { TutorialPage } from './components/TutorialPage';
import { Footer } from './components/Footer';
import { href, pages, readRoute, referencePages, tutorials, type Page, type Tutorial } from './data/pages';
import { localeTag, ui, words, type Lang } from './data/i18n';

function GuideNav({ lang, page }: { lang: Lang; page: Page }) {
  const [open,setOpen] = useState(false);
  useEffect(() => { setOpen(false); }, [page]);
  const groups = [
    { title: ui.start[lang], links: ['docs/install'] as Page[] },
    { title: ui.tutorials[lang], links: ['tutorials', ...tutorials] as Page[] },
    { title: ui.reference[lang], links: [...referencePages, 'benchmarks'] as Page[] },
  ];
  return <aside className="guide-nav"><button className="guide-nav-toggle" onClick={() => setOpen(!open)} aria-expanded={open} aria-controls="guide-links"><Menu size={17}/>{ui.docs[lang]}<ChevronDown size={16}/></button>
    <nav id="guide-links" className={open ? 'is-open' : ''} aria-label={ui.docs[lang]}>{groups.map(g => <div key={g.title}><h2>{g.title}</h2>{g.links.map(link => <a aria-current={link === page ? 'page' : undefined} href={href(lang,link)} key={link}>{pages[link].title[lang]}</a>)}</div>)}</nav></aside>;
}

export default function App() {
  const [{lang,page,path},setRoute] = useState(readRoute);
  const main = useRef<HTMLElement>(null);
  const previous = useRef(path);
  useEffect(() => {
    const sync = () => {
      const route = readRoute();
      if (location.hash !== href(route.lang,route.path)) history.replaceState(null,'',href(route.lang,route.path));
      setRoute(route);
    };
    sync();
    window.addEventListener('hashchange',sync);
    return () => window.removeEventListener('hashchange',sync);
  }, []);
  useEffect(() => {
    document.documentElement.lang = localeTag[lang];
    document.title = (page !== null ? pages[page].title[lang] : '404') + ' · fast-audio';
    if (page !== null) document.querySelector('meta[name="description"]')?.setAttribute('content',pages[page].desc[lang]);
    if (previous.current !== path) {
      main.current?.focus({preventScroll:true}); window.scrollTo(0,0); previous.current = path;
    }
  }, [lang,page,path]);

  let content;
  if (page === '') content = <><Hero lang={lang}/><StoryPillars lang={lang}/></>;
  else if (page === null) content = <div className="frame missing"><p className="eyebrow">404</p><h1>{words('Page not found','页面未找到','ページが見つかりません')[lang]}</h1><a className="text-link" href={href(lang)}>{ui.home[lang]}</a></div>;
  else {
    let body;
    switch (page) {
      case 'docs/install': body = <QuickStart lang={lang}/>; break;
      case 'docs/api': body = <CodeShowcase lang={lang} initialQuery=""/>; break;
      case 'docs/architecture': body = <ArchitectureStack lang={lang}/>; break;
      case 'docs/operators': body = <CoverageMatrix lang={lang}/>; break;
      case 'docs/roadmap': body = <RoadmapSection lang={lang}/>; break;
      case 'benchmarks': body = <BenchmarkArena lang={lang}/>; break;
      case 'tutorials': body = <GuideCards lang={lang}/>; break;
      default: body = <TutorialPage lang={lang} page={page as Tutorial}/>;
    }
    const section = page.startsWith('tutorials/') ? {label:ui.tutorials,link:'tutorials'} : {label:ui.docs,link:'docs/install'};
    content = <div className="frame docs-layout"><GuideNav lang={lang} page={page}/><article className="doc-page" key={page}>
      <nav className="breadcrumbs" aria-label={words('Breadcrumb','所在位置','パンくずリスト')[lang]}><a href={href(lang)}>{ui.home[lang]}</a><ChevronRight size={13}/>{page.startsWith('tutorials/') || page.startsWith('docs/') ? <><a href={href(lang,section.link)}>{section.label[lang]}</a><ChevronRight size={13}/></> : null}<span>{pages[page].title[lang]}</span></nav>
      <header className="doc-head"><h1>{pages[page].title[lang]}</h1><p>{pages[page].desc[lang]}</p></header>{body}
    </article></div>;
  }
  return <><Navbar lang={lang} path={path}/><main id="main-content" tabIndex={-1} ref={main}>{content}</main><Footer lang={lang}/></>;
}

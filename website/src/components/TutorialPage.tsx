import { BookOpen } from 'lucide-react';
import { lessons } from '../data/tutorials';
import { href, pages, tutorials, type Tutorial } from '../data/pages';
import { ui, words, type Lang } from '../data/i18n';
import { CodeBlock } from './CodeBlock';

export function TutorialPage({ lang, page }: { lang: Lang; page: Tutorial }) {
  const lesson = lessons[page];
  const nextPage = tutorials[tutorials.indexOf(page) + 1] ?? 'docs/api';
  return <div className="reading">
    <p>{lesson.idea[lang]}</p>
    <h2>{words('Run the example', '运行示例', 'コード例を実行')[lang]}</h2>
    <p>{words('From a fast-audio source checkout, save this file in the examples directory. The portable CPU backend is sufficient.', '在 fast-audio 源码目录中，将此文件保存到 examples 目录。使用通用 CPU 后端即可运行。', 'fast-audio のソースツリーの examples ディレクトリにこのファイルを保存します。標準 CPU バックエンドで実行できます。')[lang]}</p>
    <CodeBlock code={lesson.code} lang={lang} filename={lesson.name + '.rs'} download/>
    <CodeBlock code={'cargo run --locked --example ' + lesson.name} lang={lang} shell/>
    <h2>{words('Check the result', '检查结果', '結果を確認')[lang]}</h2><p>{lesson.check[lang]}</p>
    <div className="next-read"><small>{ui.next[lang]}</small><a className="text-link" href={href(lang,nextPage)}><BookOpen size={17}/>{pages[nextPage].title[lang]}</a></div>
  </div>;
}

export const languages = { en: 'English', zh: '简体中文', ja: '日本語' } as const;
export type Lang = keyof typeof languages;
export type Words = Record<Lang, string>;
export const words = (en: string, zh: string, ja: string): Words => ({ en, zh, ja });
export const localeTag = { en: 'en', zh: 'zh-CN', ja: 'ja' };

export const ui = {
  home: words('Home', '首页', 'ホーム'),
  docs: words('Documentation', '文档', 'ドキュメント'),
  tutorials: words('Tutorials', '教程', 'チュートリアル'),
  benchmarks: words('Benchmarks', '性能基准', 'ベンチマーク'),
  start: words('Get started', '快速上手', 'はじめる'),
  search: words('Search', '搜索', '検索'),
  searchDocs: words('Search the site', '搜索全站', 'サイト内を検索'),
  noResults: words('No matches. Try a different term.', '未找到匹配项，请换个关键词。', '一致する項目がありません。別の言葉で検索してください。'),
  close: words('Close', '关闭', '閉じる'),
  menu: words('Menu', '菜单', 'メニュー'),
  language: words('Language', '语言', '言語'),
  all: words('All', '全部', 'すべて'),
  skip: words('Skip to content', '跳到正文', '本文へ移動'),
  copy: words('Copy', '复制', 'コピー'),
  copied: words('Copied', '已复制', 'コピー済み'),
  copyError: words('Select the code to copy it manually.', '请选中代码后手动复制。', 'コードを選択して手動でコピーしてください。'),
  learn: words('Learning resources', '学习资源', '学習リソース'),
  reference: words('Reference', '参考资料', 'リファレンス'),
  next: words('Continue reading', '继续阅读', '次に読む'),
  requirements: words('Requirements', '环境要求', '必要な環境'),
  browse: words('Browse tutorials', '浏览教程', 'チュートリアルを見る'),
  tutorial: words('Tutorial', '教程', 'チュートリアル'),
  measure: words('Measurement', '性能测量', '性能測定'),
};

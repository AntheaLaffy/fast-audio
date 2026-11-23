import React from 'react';

interface Props {
  code: string;
  language?: 'rust' | 'shell';
}

const RUST_KEYWORDS = new Set([
  'use',
  'pub',
  'fn',
  'let',
  'mut',
  'for',
  'in',
  'if',
  'else',
  'match',
  'return',
  'struct',
  'enum',
  'impl',
  'trait',
  'const',
  'unsafe',
  'as',
  'true',
  'false',
  'Some',
  'None',
  'Ok',
  'Err',
]);

const RUST_TYPES = new Set([
  'Result',
  'Tensor',
  'Shared',
  'Tracked',
  'Layout',
  'Shape',
  'Builder',
  'Program',
  'Binding',
  'Value',
  'Cpu',
  'Cuda',
  'Tape',
  'BinaryOp',
  'UnaryOp',
  'ReduceOp',
  'Vec',
  'usize',
  'f32',
  'i32',
  'u64',
]);

function highlightRustLine(line: string, lineIndex: number): React.ReactNode {
  const commentIdx = line.indexOf('//');
  const codePart = commentIdx >= 0 ? line.slice(0, commentIdx) : line;
  const commentPart = commentIdx >= 0 ? line.slice(commentIdx) : '';

  const tokens: React.ReactNode[] = [];
  const regex = /("[^"]*"|\b\d+(?:\.\d+)?(?:e[+-]?\d+)?\b|[A-Za-z_][A-Za-z0-9_]*!?|#[[\w()=": ]+\]|::|->|[{}()[\],;.&*+-=<>?]+)/g;
  let lastIndex = 0;
  let match: RegExpExecArray | null;
  let k = 0;

  while ((match = regex.exec(codePart)) !== null) {
    if (match.index > lastIndex) {
      tokens.push(codePart.slice(lastIndex, match.index));
    }
    const tok = match[0];
    let cls = 'text-zinc-200';
    if (tok.startsWith('"')) {
      cls = 'text-emerald-300';
    } else if (tok.startsWith('#[')) {
      cls = 'text-amber-300';
    } else if (/^\d/.test(tok)) {
      cls = 'text-amber-300';
    } else if (tok.endsWith('!')) {
      cls = 'text-orange-400 font-medium';
    } else if (RUST_KEYWORDS.has(tok)) {
      cls = 'text-orange-400 font-medium';
    } else if (RUST_TYPES.has(tok) || /^[A-Z][A-Za-z0-9_]*$/.test(tok)) {
      cls = 'text-sky-300';
    } else if (/^[a-z_][a-z0-9_]*$/.test(tok)) {
      const nextChar = codePart[regex.lastIndex];
      if (nextChar === '(') {
        cls = 'text-amber-200';
      } else {
        cls = 'text-zinc-200';
      }
    } else {
      cls = 'text-zinc-500';
    }
    tokens.push(
      <span key={`${lineIndex}-${k++}`} className={cls}>
        {tok}
      </span>,
    );
    lastIndex = regex.lastIndex;
  }

  if (lastIndex < codePart.length) {
    tokens.push(codePart.slice(lastIndex));
  }

  if (commentPart) {
    tokens.push(
      <span key={`${lineIndex}-c`} className="text-zinc-500 italic">
        {commentPart}
      </span>,
    );
  }

  return tokens;
}

function highlightShellLine(line: string, lineIndex: number): React.ReactNode {
  const parts = line.split(/(\s+)/);
  let seenCmd = false;
  return parts.map((part, idx) => {
    if (/^\s+$/.test(part)) return part;
    if (part.includes('=') && !seenCmd) {
      const [k, v] = part.split('=');
      return (
        <span key={`${lineIndex}-${idx}`}>
          <span className="text-sky-300">{k}</span>
          <span className="text-zinc-500">=</span>
          <span className="text-amber-300">{v}</span>
        </span>
      );
    }
    if (!seenCmd) {
      seenCmd = true;
      return (
        <span key={`${lineIndex}-${idx}`} className="text-orange-400 font-medium">
          {part}
        </span>
      );
    }
    if (part.startsWith('--') || part.startsWith('-')) {
      return (
        <span key={`${lineIndex}-${idx}`} className="text-sky-300">
          {part}
        </span>
      );
    }
    return (
      <span key={`${lineIndex}-${idx}`} className="text-zinc-200">
        {part}
      </span>
    );
  });
}

export const SyntaxHighlight: React.FC<Props> = ({ code, language = 'rust' }) => {
  const lines = code.split('\n');
  return (
    <pre className="overflow-x-auto text-[12.5px] leading-[1.65] font-mono">
      <code>
        {lines.map((line, idx) => (
          <div key={idx} className="table-row hover:bg-white/[0.025]">
            <span className="table-cell select-none pr-4 text-right text-zinc-600 w-9">
              {idx + 1}
            </span>
            <span className="table-cell whitespace-pre">
              {language === 'rust' ? highlightRustLine(line, idx) : highlightShellLine(line, idx)}
            </span>
          </div>
        ))}
      </code>
    </pre>
  );
};

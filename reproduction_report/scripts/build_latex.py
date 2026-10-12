"""Compile the requested Markdown documents into Chinese PDF through local XeLaTeX."""
from pathlib import Path
import json
import re
import shutil
import subprocess
import sys
import os

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'tools' / 'latex'
LATEX = Path(r'D:\Program Files\texlive\2025\bin\windows\xelatex.exe')


def plain_escape(text):
    replacements = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$',
                    '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}',
                    '~': r'\textasciitilde{}', '^': r'\textasciicircum{}'}
    return ''.join(replacements.get(character, character) for character in text)


def breakable_code(text):
    chunks = []
    for index, character in enumerate(text):
        chunks.append(plain_escape(character))
        if character in '/\\_.:-=' or (index + 1) % 12 == 0:
            chunks.append(r'\allowbreak{}')
    return r'\texttt{' + ''.join(chunks) + '}'


def inline(text):
    pattern = re.compile(r'(`[^`]+`|\*\*[^*]+\*\*|\[[^]]+\]\([^)]*\))')
    out = []
    last = 0
    for match in pattern.finditer(text):
        out.append(plain_escape(text[last:match.start()]))
        token = match.group()
        if token.startswith('`'):
            out.append(breakable_code(token[1:-1]))
        elif token.startswith('**'):
            out.append(r'\textbf{' + plain_escape(token[2:-2]) + '}')
        else:
            label, url = re.match(r'\[([^]]+)\]\((.*)\)', token).groups()
            if url.startswith(('http://', 'https://')):
                out.append(r'\href{' + plain_escape(url) + '}{' + plain_escape(label) + '}')
            else:
                out.append(plain_escape(label) + '（' + breakable_code(url) + '）')
        last = match.end()
    out.append(plain_escape(text[last:]))
    return ''.join(out)


def convert(markdown_path):
    lines = markdown_path.read_text(encoding='utf-8-sig').splitlines()
    title = next((line[2:].strip() for line in lines if line.startswith('# ')), markdown_path.stem)
    output = []
    index = 0
    list_kind = None

    def close_list():
        nonlocal list_kind
        if list_kind:
            output.append(r'\end{' + list_kind + '}')
            list_kind = None

    while index < len(lines):
        line = lines[index].strip()
        if not line:
            close_list(); output.append(''); index += 1; continue
        if line.startswith('# '):
            index += 1; continue
        if line.startswith('```'):
            close_list(); index += 1; code = []
            while index < len(lines) and not lines[index].strip().startswith('```'):
                code.append(lines[index]); index += 1
            output.append(r'\begin{Verbatim}[breaklines=true,breakanywhere=true,fontsize=\scriptsize,frame=single]')
            output.extend(code); output.append(r'\end{Verbatim}'); index += 1; continue
        image = re.fullmatch(r'!\[([^]]*)\]\(([^)]+)\)', line)
        if image:
            close_list()
            caption, relative = image.groups()
            source = (markdown_path.parent / relative).resolve()
            if not source.is_file():
                raise FileNotFoundError(source)
            graphic = Path(os.path.relpath(source, OUTPUT)).as_posix()
            output.extend([r'\begin{figure}[H]', r'\centering',
                r'\includegraphics[width=0.98\linewidth,height=0.66\textheight,keepaspectratio]{\detokenize{' + graphic + '}}',
                r'\caption*{' + inline(caption or source.stem) + '}', r'\end{figure}'])
            index += 1; continue
        heading = re.match(r'^(#{2,4})\s+(.+)$', line)
        if heading:
            close_list(); level = len(heading.group(1))
            command = {2:'section', 3:'subsection', 4:'subsubsection'}[level]
            title = re.sub(r'^\d+[.、]\s*', '', heading.group(2))
            output.append('\\' + command + '{' + inline(title) + '}')
            index += 1; continue
        if line.startswith('|'):
            close_list(); rows = []
            while index < len(lines) and lines[index].strip().startswith('|'):
                cells = [cell.strip() for cell in lines[index].strip().strip('|').split('|')]
                if not all(re.fullmatch(r'[:\s-]+', cell) for cell in cells):
                    rows.append(cells)
                index += 1
            columns = max(map(len, rows))
            output.append(r'\begin{longtable}{' + '*{' + str(columns) + '}{>{\\raggedright\\arraybackslash}p{' + f'{.91/columns:.4f}' + r'\linewidth}}}')
            output.append(r'\toprule')
            for row_index, cells in enumerate(rows):
                row = cells + ['']*(columns-len(cells))
                output.append(' & '.join(inline(cell) for cell in row) + r' \\')
                if row_index == 0:
                    output.append(r'\midrule')
            output.extend([r'\bottomrule',r'\end{longtable}']); continue
        bullet = re.match(r'^[-*]\s+(.+)$', line)
        numbered = re.match(r'^\d+[.)]\s+(.+)$', line)
        if bullet or numbered:
            kind = 'itemize' if bullet else 'enumerate'
            if list_kind != kind:
                close_list(); output.append(r'\begin{' + kind + '}'); list_kind = kind
            output.append(r'\item ' + inline((bullet or numbered).group(1)))
            index += 1; continue
        close_list()
        if line.startswith('>'):
            output.append(r'\begin{quote}' + inline(line.lstrip('> ').strip()) + r'\end{quote}')
        elif re.fullmatch(r'[-*_]{3,}',line):
            output.append(r'\medskip\hrule\medskip')
        else:
            output.append(inline(line))
        index += 1
    close_list()
    preamble = r'''\documentclass[UTF8,11pt,a4paper,fontset=windows]{ctexart}
\usepackage[margin=22mm]{geometry}
\usepackage{graphicx,float,booktabs,longtable,array,enumitem,fvextra,xurl,hyperref,fancyhdr,caption}
\setmonofont{Consolas}
\hypersetup{colorlinks=true,urlcolor=blue,linkcolor=black}
\setlist{itemsep=2pt,topsep=3pt,leftmargin=2em}
\setlength{\parskip}{4pt}
\setlength{\tabcolsep}{3pt}
\setlength{\headheight}{16pt}
\setlength{\emergencystretch}{3em}
\pagestyle{fancy}\fancyhf{}\fancyhead[L]{DTIUIE复现档案}\fancyhead[R]{官方代码默认配置}\fancyfoot[C]{\thepage}
\title{''' + inline(title) + r'''}
\author{从真实训练日志、TensorBoard与GPU巡检重建}
\date{训练日期：2026年10月3日}
\begin{document}
\maketitle
\tableofcontents
\clearpage
'''
    return preamble + '\n'.join(output) + '\n\\end{document}\n'


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    targets = [('复现过程与结果.md','DTIUIE_reproduction_report','DTIUIE_复现报告.pdf'),
               ('复现交接文档.md','DTIUIE_handover','DTIUIE_交接文档.pdf')]
    metadata = []
    for markdown, base, public_pdf in targets:
        source = ROOT/'docs'/markdown
        tex = OUTPUT/(base+'.tex')
        tex.write_text(convert(source), encoding='utf-8')
        for pass_number in (1,2):
            result = subprocess.run([str(LATEX),'-interaction=nonstopmode','-halt-on-error',tex.name],
                        cwd=OUTPUT, capture_output=True, text=True, encoding='utf-8', errors='replace')
            (OUTPUT/(base+f'.pass{pass_number}.stdout.log')).write_text(result.stdout,encoding='utf-8')
            if result.returncode:
                print(result.stdout[-6000:]);raise RuntimeError('XeLaTeX compilation failed: '+base)
        pdf = OUTPUT/(base+'.pdf')
        assert pdf.is_file() and pdf.stat().st_size>10000
        shutil.copyfile(pdf,ROOT/public_pdf)
        metadata.append({'markdown':str(source),'tex':str(tex),'pdf':str(pdf),
                         'public_pdf':public_pdf,'engine':str(LATEX),'passes':2,'exit_code':0})
        print('Compiled:',pdf)
    (OUTPUT/'latex_execution.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()

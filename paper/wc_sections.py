#!/usr/bin/env python3
"""Per-section prose word counts for main.tex, excluding float environments."""
import re, sys
path = sys.argv[1] if len(sys.argv) > 1 else 'main.tex'
lines = open(path).read().split('\n')
marks = []
for i, l in enumerate(lines):
    if re.match(r'\\(sub)?section\{', l):
        name = re.sub(r'\\label\{.*?\}', '', l).strip()
        name = re.sub(r'\\texorpdfstring\{(.*?)\}\{.*?\}', r'\1', name)
        marks.append((i, name[:60]))
    elif l.startswith(r'\begin{abstract}'):
        marks.append((i, 'ABSTRACT'))
    elif l.startswith(r'\appendix'):
        marks.append((i, '--- APPENDIX ---'))
    elif l.startswith(r'\begin{acknowledgements}'):
        marks.append((i, 'ACKNOWLEDGEMENTS'))
marks.append((len(lines), 'END'))
main_tot = app_tot = 0
in_app = False
for (a, name), (b, _) in zip(marks, marks[1:]):
    if name.startswith('---'):
        in_app = True
        print(f"{'':62s} {'':>6s}")
        continue
    chunk = '\n'.join(lines[a:b])
    prose = re.sub(r'\\begin\{(figure\*?|table\*?)\}.*?\\end\{\1\}', '', chunk, flags=re.S)
    w = len(prose.split())
    if in_app: app_tot += w
    elif name not in ('ABSTRACT', 'ACKNOWLEDGEMENTS'): main_tot += w
    print(f"{name:62s} {w:6d}   lines {a+1}-{b}")
print(f"\n{'MAIN TEXT PROSE':62s} {main_tot:6d}")
print(f"{'APPENDIX PROSE':62s} {app_tot:6d}")

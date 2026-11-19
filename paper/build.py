"""Build the paper from the analysis outputs, refusing to if it quotes anything they do not make.

main.tex `\\input`s its macro files and `\\includegraphics` its figures by bare
name. Every such file is made by an output of astra.yaml (`numbers`,
`scalar_numbers`, ..., `fig_*`), except `legacy_numbers.tex`, which is kept by
hand beside main.tex. This script puts paper/ and the output directories it
is given on TeX's search path and compiles main.tex in `--out` as paper.pdf.

The gate, before and during the build:

  * every `\\input` main.tex names must exist in paper/ or in one of the given
    directories, and every figure in one of the given directories, each name
    in exactly one place, so no copy in paper/ can shadow an output;
  * pdflatex runs with -halt-on-error, so a macro that no `\\input` file
    defines stops the build ("Undefined control sequence"), and the log is
    checked for undefined references and citations.

It also lists the macros the given numbers files define that main.tex never
quotes, which is not an error.

Run:  python3 paper/build.py --out <dir> results/fiducial/numbers results/fiducial/fig_shear_leakage ...
      (any directories: `results/fiducial/*/` works)
Out:  <out>/paper.pdf and the LaTeX auxiliaries beside it.
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parent


def body(tex):
    """The text of a .tex file with comments removed."""
    return re.sub(r"(?<!\\)%.*", "", tex)


def locate(name, dirs):
    """The one directory among `dirs` that holds `name`."""
    found = [d for d in dirs if (d / name).is_file()]
    if len(found) != 1:
        where = ", ".join(map(str, found)) or "nowhere"
        raise SystemExit(f"gate: {name} found in {where}; main.tex needs it in exactly one place")
    return found[0]


def gate(dirs):
    """Check main.tex's inputs and figures; return the macros the inputs define but main.tex never uses."""
    main = body((PAPER / "main.tex").read_text())
    defined = {}
    for name in re.findall(r"\\input\{([^}]+)\}", main):
        name = name if name.endswith(".tex") else name + ".tex"
        path = locate(name, [PAPER, *dirs]) / name
        for macro in re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", path.read_text()):
            defined[macro] = name
    for name in re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", main):
        if locate(name, [PAPER, *dirs]) == PAPER:
            raise SystemExit(f"gate: figure {name} is in paper/, not made by an output")
    used = set(re.findall(r"\\([A-Za-z]+)", main))
    return sorted((m, f) for m, f in defined.items() if m not in used)


def latex(out, env, *args):
    run = subprocess.run(args, cwd=out, env=env, capture_output=True, text=True)
    if run.returncode:
        log = (out / "paper.log").read_text(errors="replace") if args[0] == "pdflatex" else run.stdout
        lines = log.splitlines()
        # Each TeX error and the two lines of context that name the culprit.
        errors = [line for i, line in enumerate(lines) if line.startswith("!")
                  or any(lines[k].startswith("!") for k in (i - 1, i - 2) if k >= 0)]
        raise SystemExit(f"gate: {args[0]} failed\n" + "\n".join(errors or lines[-30:]))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("dirs", type=Path, nargs="+", help="output directories to search")
    args = ap.parse_args()
    dirs = [d.resolve() for d in args.dirs]

    for macro, name in gate(dirs):
        print(f"unquoted: \\{macro} ({name})")

    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out.resolve()
    search = os.pathsep.join(map(str, [PAPER, *dirs])) + os.pathsep
    env = dict(os.environ, TEXINPUTS=search, BIBINPUTS=search, BSTINPUTS=search)
    pdflatex = ("pdflatex", "-halt-on-error", "-interaction=nonstopmode", "-jobname=paper",
                str(PAPER / "main.tex"))
    latex(out, env, *pdflatex)
    latex(out, env, "bibtex", "paper")
    for i in range(5):
        latex(out, env, *pdflatex)
        log = (out / "paper.log").read_text(errors="replace")
        if i >= 1 and "Rerun to get" not in log:
            break

    problems = [line for line in log.splitlines()
                if re.search(r"(Reference|Citation) .* undefined", line)]
    if problems:
        raise SystemExit("gate: " + "\n".join(problems))
    print(f"wrote {out / 'paper.pdf'}")


if __name__ == "__main__":
    sys.exit(main())

# Editing the paper

`main.tex` is built by the `paper` output of `astra.yaml` (`lc run paper -u fiducial`), which runs `build.py` and writes `results/fiducial/paper/paper.pdf`.

**Every measured number is a macro.**
The prose never types a measured value.
Each number is a LaTeX macro written by one of the `*_numbers` outputs: `analysis/paper_numbers/main.py` writes `numbers.tex`, and `analysis/paper_numbers/<name>_numbers.py` writes `<name>_numbers.tex`.
To quote a new number, add it to the writer that owns that analysis, rerun that output, and cite the macro.
Code docstrings and comments name the macro rather than a value that would go stale.
When the prose stops quoting a macro, drop it from its writer; `build.py` lists the defined macros that `main.tex` never uses.

**The build is the gate.**
`build.py` refuses to compile if `main.tex` inputs a file or includes a figure that no output makes, or if a copy in `paper/` would shadow an output.
It runs pdflatex with `-halt-on-error`, so an undefined macro stops the build, and it fails on undefined references or citations.

**Editing by script.**
Patch `main.tex` with exact old-to-new pairs that each match once.
After editing, rebuild and check the log for `Overfull` boxes before committing.

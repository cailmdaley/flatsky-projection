#!/usr/bin/env python3
"""Check every TeX ``quotation'' against pinned, stored source text.

Run from any directory: python analysis/check_paper_quotes.py
A nonzero exit means a changed/unregistered quote, missing source, or mismatch.
Only typography is normalized; words, case and punctuation must match.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "paper/citation_audit"


def normalize(text):
    text = text.replace("\\ell", "ℓ")
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\\(?:textit|textbf|mathrm)\{([^{}]*)\}", r"\1", text)
    text = text.replace("$", "").replace("{", "").replace("}", "")
    text = text.translate(str.maketrans({"−": "-", "–": "--", "—": "---", "’": "'", "‘": "'", "\u00ad": ""}))
    # PDF line-wrap hyphens are not lexical hyphens. Preserve all other hyphens.
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)
    return " ".join(text.split())


def check(tex, manifest, audit=AUDIT):
    errors = []
    records = {r["quote"]: r for r in manifest}
    quotes = re.findall(r"``(.*?)''", tex, re.S)
    for quote in quotes:
        line = tex[:tex.index("``" + quote + "''")].count("\n") + 1
        record = records.get(quote)
        if record is None:
            errors.append(f"line {line}: unregistered quotation: {quote}")
            continue
        source = audit / record["source"]
        if not source.exists():
            errors.append(f"line {line}: missing source {source}")
            continue
        raw = source.read_bytes()
        if hashlib.sha256(raw).hexdigest() != record["sha256"]:
            errors.append(f"line {line}: stored source hash changed: {source}")
            continue
        if normalize(quote) not in normalize(raw.decode()):
            errors.append(f"line {line}: {record['key']} quotation does not match source: {quote}")
    return quotes, errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tex", type=Path, default=ROOT / "paper/main.tex")
    args = parser.parse_args()
    manifest = json.loads((AUDIT / "quotes.json").read_text())
    quotes, errors = check(args.tex.read_text(), manifest)
    for error in errors:
        print("FAIL:", error)
    print(f"{len(quotes)} quotations checked; {len(errors)} failures")
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())

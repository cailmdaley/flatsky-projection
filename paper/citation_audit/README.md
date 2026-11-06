# Citation audit

The audit of `paper/main.tex` at `5fec14a` covers 79 distinct reference keys and all ten direct quotations.
The sentence-by-sentence verdict table lives in the Felt fiber `science/spt/pol-project-paper/citation-audit`.
The paper owner applies the prose fixes below; this audit does not edit `main.tex` or the shared bibliography.

## Required repairs

1. **DES orthographic mapping cites the wrong paper.**
   `jeffrey21` is Jeffrey, Alsing & Lanusse's DES SV neural-compression paper (arXiv:2009.08459), which uses gnomonic simulation cutouts.
   The orthographic statement is in Jeffrey et al., *DES Year 3 results: Curved-sky weak lensing mass map reconstruction* (arXiv:2105.13539, MNRAS 505, 4626, DOI 10.1093/mnras/stab1495), §3.4.
   It applies specifically to **Glimpse**, not to all DES reconstruction methods; the other methods are spherical.
   Replace “on orthographic tangent planes in DES” with “on orthographic tangent planes in the DES Glimpse reconstruction”, and cite a new `jeffrey21massmaps` entry.
   `jeffrey_massmaps.bib` is the ADS-exported entry to add, with its key renamed.
   The source text is stored in `sources/jeffrey_massmaps.txt`.
2. **The 0.3–1% figure is for a theory calculation, not a universal transform-error bound.**
   At the sentence beginning “A mask boundary leaves some modes ambiguous”, replace “the flat-sky approximation to the harmonic transform adds an error controlled at the 0.3–1% level” with “the flat-sky calculation of lensed CMB spectra differs from the curved-sky calculation by 0.3–1%”.
   Cite `challinor05` for that number.
   Its abstract and §IV give it for non-perturbative lensed spectra, with different errors for temperature and polarization.
   `bernardeau11` supports a separate theory-side statement about controlled approximations (including restrictions on the ISW contribution), not a bound on projecting arbitrary observed maps.
   Drop that reference from this numerical clause, or give it its own accurately scoped clause.
3. **Zhou et al. do not fit the amplitude of their leakage template.**
   At “The template is a projection-specific case of the E-only template subtraction of Zhou et al.”, replace “with an amplitude fixed by the projection rather than fitted” with “with its response fixed by the projection geometry”.
   `zhou25` directly subtracts a propagated satellite E-mode map; template fitting is an alternative discussed in their conclusion, not their demonstrated method.
   Calling our correction an analogue of their E-only template subtraction is supported.
4. **Baxter et al. use a joint likelihood, not a stacking estimator.**
   Replace “Stacking analyses cut flat stamps around known objects” with “Cluster and halo CMB-lensing analyses cut flat stamps around known objects”.
   This retains all three citations and accurately covers Baxter's maximum-likelihood analysis as well as the two stacking examples.
5. **The Wallis quotation silently changes capitalization.**
   The abstract says “The stereographic projection, which is conformal and so preserves local angles, is the most effective planar projection.”
   Capitalize `The` inside the quotation, or paraphrase without quotation marks.
   All words and punctuation in the other nine quotations match the stored sources after typographic normalization.
6. **The `bk7` author initials are shortened relative to the source.**
   Change `{Ade}, P.` to `{Ade}, P.~A.~R.` and `{Kovac}, J.` to `{Kovac}, J.~M.` in this entry only.
   Its DOI, title, year, volume and article number are correct.
   Collaboration-first versus individual-first author ordering is a style choice, not a wrong-paper defect.
7. **“Most experiments” is not established by the cited examples.**
   Replace “As analyses have pushed to wider areas, most experiments have switched to SHTs” with “Several wide-area analyses now use SHTs”.
   The cited papers establish the named pipelines, not a census of experiments.

## Verification and metadata

`ads.json` stores the ADS search records for all 79 cited keys (78 ADS matches; OSPool is a service DOI, verified separately against DataCite).
`ads-canonical.bib` stores ADS exports for the 78 matches.
I compared titles, author lists, publication years, volumes, first pages/article numbers and DOIs against the current bibliography.
Author-list differences other than the two `bk7` initials are collaboration placement, the intentional “and others” on `osg07`, and ADS dropping the accent on Umiltà; these do not require a change.
Two misleading ADS fields must **not** be copied blindly:

- `osg09` has page “92” in ADS, but the DOI's publisher-deposited Crossref record says **428–432**, matching the existing bibliography (`osg09.json`).
- `snyder87` is USGS Professional Paper **1395**, not a journal article on page 27; the existing book/series representation agrees with the USGS report.

Primary texts for 77 references were retrieved, including the USGS manual and Kaiser–Squires scan.
`osg09` was verified through its DOI metadata and `ospool` through its service page/DataCite record rather than reading an inaccessible conference PDF.
These are acknowledgements, not evidence for scientific claims.
The large retrieval cache, including all scientific-source PDFs/text, is `/sptlocal/user/cailmdaley/pol_project/citation_audit/`.
Only the sources needed by the quotation gate and the corrected DES reference are checked into this directory.

## Repeatable quotation gate

```bash
python analysis/check_paper_quotes.py
python -m unittest discover -s analysis -p test_paper_quotes.py -v
```

The gate checks every LaTeX double-backtick quotation in the **current** manuscript against stored primary-source text, and rejects unregistered quotations, changed source hashes, and non-verbatim text.
It normalizes Unicode ligatures, math delimiters, TeX `\ell`, whitespace, typographic dashes and PDF line-wrap hyphens; it does not normalize case or rewrite words/punctuation.
`quotes.json` records source URLs and SHA-256 hashes.
The D1 source is the `skymaps.tex` member of the arXiv source archive: PDF extraction puts a full-page figure in the middle of that sentence, so a PDF substring search would falsely reject it.
Other quote sources were converted with `pdftotext` without `-layout` so columns remain in reading order.
The lowercase Wallis text remains registered to expose the original mismatch; the uppercase repair is registered too.

**Observed result before the owner applies the repair:** ten quotations, one failure (Wallis capitalization).
All five regression tests pass, including the real-source check with the recommended capitalization, a fabricated but registered quotation, and source tampering.
This verifies textual fidelity, not whether a sentence faithfully interprets its source; the human-readable audit covers that separate question.
Run the gate again after the paper-restructure integration and register any genuinely new quotations against their sources rather than bypassing it.

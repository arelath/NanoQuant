# Publication Draft

This folder contains an arXiv-oriented draft of the NanoQuant rewrite research.

## Files

- `main.tex`: complete paper draft, including the main method, current results,
  algorithm catalog, negative results, and reproducibility appendix.
- `references.bib`: working bibliography. Entries explicitly marked
  `PLACEHOLDER` must be verified before submission.
- `claim-status.md`: claim-by-claim evidence and placeholder checklist.
- `preview.html`: Pandoc-generated browser preview of the current draft.

## Build

From this directory:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Regenerate the browser preview with:

```powershell
pandoc main.tex --standalone --citeproc --mathjax `
  --bibliography=references.bib `
  --metadata title="Beyond Matrix Reconstruction" `
  -o preview.html
```

The source uses the standard `article` class so the draft remains portable to
arXiv. A venue-specific template can replace the document class later without
changing the scientific content.

The initial draft was parsed successfully by Pandoc with all citation keys
resolved. A TeX distribution was not available in the authoring environment,
so the PDF build commands above still need to be run on a machine with LaTeX.

## Placeholder policy

Red `[PLACEHOLDER: ...]` text identifies missing evidence, metadata, figures,
or conclusions. Placeholders are intentionally compileable and should not be
silently replaced with extrapolated values. In particular, the abstract and
conclusion must not claim cross-model or combined-method results until those
runs are complete.

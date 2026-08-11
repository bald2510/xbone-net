# XBone-Net ACIIDS full-paper draft

This folder contains an English Springer LNCS/LNAI manuscript prepared as an ACIIDS full-paper draft.

## Build

Run BibTeX between the first and later PDFLaTeX passes:

```text
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The official Springer `llncs.cls` and `splncs04.bst` files are included so that the folder is self-contained.

## Submission items that still require author confirmation

- Final author order, corresponding author, and contact email.
- Exact institutional naming required by each author.
- Ethics or institutional approval identifier, if one exists and is applicable.
- Funding, acknowledgements, data-provider wording, and conflict-of-interest statement.
- Whether ACIIDS requests author-identifying or anonymized initial submission for the selected track.
- Final language edit and similarity check against the thesis and any other manuscripts.

## Evidence policy

The numerical results in `main.tex` are copied from the verified three-seed summaries already used in the thesis. CTCH is treated as the primary clinical-history dataset. BTXRD is explicitly secondary because its auxiliary text is generated from metadata and can contain label-correlated information. The manuscript does not claim state-of-the-art performance, clinical deployment readiness, lesion-localization validity, or reliable automatic OOD rejection.

# XBone-Net: Parameter-Efficient Radiograph Classification with Clinical History

Open `main.tex` as the main document and select **pdfLaTeX**. The source archive can be imported directly into Overleaf. Keep the supplied `.tex`, `.bib`, and `.dat` files together.

Local build with a TeX distribution containing IEEEtran, TikZ and PGFPlots:

```text
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

The released PDF is six A4 pages, including references, using PDF 1.6 and the standard IEEE conference font size. The bibliography column break is controlled by `\IEEEtriggeratref{10}`; adjust it after substantial text changes.

- `classification_table.tex`, `ablation_table.tex`: audited three-seed results, in percent.
- `framework.tex`: editable vector diagram of offline adaptation and online inference.
- `ig_curves.tex`, `ig_image.dat`, `ig_text.dat`: vector plots from all 669 CTCH test cases, seed 42. Curves show means; no confidence band is claimed.
- `references.bib`: bibliography.
- `evidence/audit.json`: aggregate evidence with input file hashes.
- `REVIEW_NOTES_VI.md`: changes, evidence and items requiring author verification.

The source package does not contain patient records, predictions, trained models, or a training environment. Results were recomputed from existing local artifacts, except the BTXRD Image-only aggregate supplied by the author in the follow-up request. Its provenance is recorded separately in `evidence/author_supplied_results.json`; it has not been independently recomputed from aligned per-seed predictions. No additional training was performed.

In the updated tables, bold numbers indicate the best mean and underlining indicates the second-best mean. Proposed configuration names are bold. Image-only appears in Table IV and is omitted from Table VI to avoid duplication. `TITLE_AND_EVIDENCE_VI.md` provides title options and an assessment of the evidence and the role of IG.

The original files supplied in Downloads were not overwritten. This revision has not been submitted to the conference.

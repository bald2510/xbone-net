# RIVF XBone-Net: qualitative IG revision

This revision adds Figure 4 and a case-level discussion to the six-page language-revised manuscript. All classification/ablation table entries, resource measurements, and full-cohort IG curve data are unchanged.

The author-supplied figure was found at `docs/report/images/chapter4/multimodal_ig_3563.png`. Case `3563_img-27903-00001.jpg` is present in the seed-42 full-test artifact with ground-truth class 4, predicted class 4, and probability 0.5004590153694153, matching the original figure's 50.0% label. The full artifact contains 669 cases.

`ig_case_source.png` is an unchanged copy of the thesis figure. `ig_case.tex` uses PDF clipping paths to display its original radiograph, original heatmap, and complete text-attribution rows. Only the document layout, line breaks, and English panel headings are changed. No image generation, replacement heatmap, or synthetic attribution was used. The example is author-selected and is not claimed to be representative of all test cases.

The figure renderer in the repository is `tools/visualize/render_multimodal_ig.py`, which uses the common benchmark IG functions. Image relevance uses the positive part of channel-summed attribution with an absolute-magnitude fallback when the positive map is degenerate. Text colors display normalized signed scores. The retained screenshot was not recomputed in this editing task. The case prediction and label were checked against the saved full-test record.

Files:

- `main.tex`: complete manuscript, six A4 pages when compiled with PDFLaTeX.
- `modality_contribution.tex`: the complete updated subsection including both IG figures, for reuse in the author's source project.
- `ig_case.tex` and `ig_case_source.png`: editable layout with the immutable original panels.
- `ig_case_figure.tex` and `ig_case_figure.pdf`: standalone English figure.
- `ig_provenance.json`: case identity, recorded prediction, source, and clipping coordinates.
- `quality_check.json`: page count, font, numerical-preservation and rendering checks.

The quantitative plot height was reduced from 47 to 38 mm without changing its data or label sizes. Repeated IG interpretation and the conclusion were shortened to retain six pages. Both Methods diagrams, all six tables, and the quantitative IG plot remain. Figure 4 is qualitative evidence of input sensitivity, not lesion-localization validation or proof of region-word alignment.

Compile `main.tex` with PDFLaTeX, BibTeX, PDFLaTeX, PDFLaTeX. A prebuilt `main.bbl` is included. This source continues the preceding reconstruction from the author's PDF and older LaTeX source, not an exact copy of their current Overleaf project.

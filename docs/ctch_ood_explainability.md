# CTCH proposed-model OOD and explainability

The analysis pipeline is post-hoc and locked to:

- experiment: `ctch/proposed/ours_xbone_net`
- checkpoints: `seed_42`, `seed_123`, and `seed_456`
- checkpoint file: `best_phase2.pth`

It never calls `train.py`. Each run validates the Phase-2 architecture,
22-class empirical-centroid head, parameter fingerprint, checkpoint SHA-256,
and feature provenance.

## Run the registered protocol

```powershell
python tools/run_ctch_analysis.py
```

The OOD protocol fits density estimators on CTCH train, calibrates thresholds on
CTCH validation ID only, and evaluates on CTCH test versus:

1. CTCH semantic OOD;
2. FracAtlas domain OOD (primary analysis uses visual-global features);
3. cross-class report mismatch;
4. same-class report mismatch (secondary control).

The explainability protocol performs stratified batch analysis with integrated
gradients, visual and clinical-token attribution, branch ablation,
visual and clinical-token deletion/insertion controls, stability,
classifier-randomization sanity checks,
and full-dimensional representation/clustering metrics.

Useful subsets:

```powershell
python tools/run_ctch_analysis.py --seeds 42 --analyses ood --ood-scenarios domain_ood
python tools/run_ctch_analysis.py --seeds 42 --analyses explainability
python tools/run_ctch_analysis.py --feature-only
python tools/run_ctch_analysis.py --table
```

Use `--overwrite` to recompute verified outputs. Otherwise feature, OOD, and
explainability artifacts are resumed only when their provenance matches.

## CTCH-OOD data coverage

Official semantic-OOD evaluation is fail-closed: every `ctch-ood.csv` row must
have an image, English X-ray report, and English clinical report. Check coverage
before starting analysis with `--step validate`; the command reports ID and OOD
image/report coverage separately and stops when either OOD modality is missing.

After restoring the source workbook and source images, repair only artifacts for
the already locked `ctch-ood.csv` cohort:

```powershell
python data/CTCH/preprocess_ctch.py --step repair-ood --input-xlsx <source.xlsx> --images-src <images_no_implants>
python data/CTCH/preprocess_ctch.py --step validate
```

`repair-ood` never regenerates `ctch-labels.csv`, `ctch-split.csv`, or
`ctch-ood.csv`. It materializes only missing OOD images/reports, translates only
missing OOD English reports, and verifies that the ID-manifest SHA-256 values
remain unchanged. Install `deep-translator` in the preprocessing environment if
uncached Vietnamese report segments still require translation.

For debugging only, incomplete coverage can be explicitly acknowledged:

```powershell
python tools/run_ctch_analysis.py --allow-incomplete-ood
```

Those artifacts record their incomplete coverage and are marked exploratory.

## Outputs

Per seed:

```text
results/ctch/proposed/ours_xbone_net/seed_<seed>/analysis/
  features/*.npz
  ood/<scenario>/ood_metrics.json
  ood/<scenario>/ood_scores.npz
  explainability/summary.json
  explainability/samples.json
  explainability/representation_overview.png
  explainability/visualizations/*.png
```

Cross-seed aggregation is written to:

```text
results/ctch/proposed/ours_xbone_net/analysis/aggregated_results.json
```

Plot an existing OOD score archive without refitting or recalibrating it:

```powershell
python tools/visualize_ood.py --scores <scenario-dir>/ood_scores.npz --output <scenario-dir>/score_distributions.png
```

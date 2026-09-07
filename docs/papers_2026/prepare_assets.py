from pathlib import Path
import csv, json, shutil
import numpy as np

BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[1]
data=list(csv.DictReader((BASE/'evidence/classification_aggregate.csv').open()))
def row(ds,name): return next(r for r in data if r['dataset']==ds and r['model']==name)
def cells(r):
    return ' & '.join(f'${float(r[k+"_mean"]):.4f}\\pm{float(r[k+"_sd"]):.4f}$' for k in ['accuracy','balanced_accuracy','f1_macro'])
for venue in ['soict','rivf']:
    out=BASE/venue
    out.mkdir(exist_ok=True)
    shutil.copy(ROOT/'docs/paper/references.bib',out/'references.bib')
    with (out/'references.bib').open('a',encoding='utf-8') as f:
        f.write(r'''
@article{BIOMEDICA,
 author={Lozano, Alejandro and Sun, Min Woo and Burgess, James and others},
 title={{BIOMEDICA}: An Open Biomedical Image-Caption Archive, Dataset, and Vision-Language Models Derived from Scientific Literature},
 journal={arXiv preprint arXiv:2501.07171}, year={2025}, doi={10.48550/arXiv.2501.07171}}
''')
    rows=[]
    for ds in ['ctch','btxrd']:
        rows.append(r'\textit{'+ds.upper()+(': $N=669$' if ds=='ctch' else ': $N=750$')+r'} & & & \\')
        for name in ['ResNet-50','DenseNet-121','CLIP','MedCLIP','PubMedCLIP','BiomedCLIP','XBone-Net']:
            rows.append(name+' & '+cells(row(ds,name))+r'\\')
        rows.append(r'\midrule')
    (out/'classification_rows.tex').write_text('\n'.join(rows[:-1]),encoding='utf-8')
    (out/'ablation_rows.tex').write_text('\n'.join(name+' & '+cells(row('ctch',name))+r'\\' for name in ['XBone-Net','Phase 2 only','Concatenation','Image only']),encoding='utf-8')
    (out/'framework.tex').write_text(r'''
\begin{tikzpicture}[font=\small,
 box/.style={draw,rounded corners=1pt,align=center,minimum height=10mm,text width=35mm,inner sep=3pt},
 arr/.style={-{Latex[length=1.8mm]},thick}]
\node[box] (data) at (0,0) {Training pairs\\image, history, label};
\node[box] (p1) at (4.1,0) {Phase 1\\class-aware alignment};
\node[box] (p2) at (8.2,0) {Phase 2\\fusion + classification};
\draw[arr] (data)--(p1); \draw[arr] (p1)--(p2);
\node[box] (new) at (0,-2.0) {New case\\image + history};
\node[box] (enc) at (4.1,-2.0) {Adapted encoders\\global + local tokens};
\node[box] (pred) at (8.2,-2.0) {Cross-attention\\linear head + softmax};
\draw[arr] (new)--(enc); \draw[arr] (enc)--(pred);
\draw[arr,dashed] (p1)--node[right,font=\scriptsize]{initialize}(enc);
\draw[arr,dashed] (p2)--node[right,font=\scriptsize]{deploy best model}(pred);
\node[font=\small\bfseries,anchor=west] at (-1.8,0.9) {OFFLINE: fit on training data; select on validation data};
\node[font=\small\bfseries,anchor=west] at (-1.8,-3.0) {ONLINE: fixed parameters; no label input or gradient update};
\end{tikzpicture}
''',encoding='utf-8')
    if venue=='soict':
        for name in ['llncs.cls','splncs04.bst']:
            shutil.copy(BASE/'template'/name,out/name)

# Validate pre-existing paired-statistics means against newly recomputed predictions.
checks=[]
for ds in ['ctch','btxrd']:
    p=ROOT/f'results/summary/classification/statistics_{ds}_xbone_vs_baselines/paired_baseline_statistics_compact.csv'
    for r in csv.DictReader(p.open(encoding='utf-8-sig')):
        k=r['metric']
        if k not in ['accuracy','balanced_accuracy','f1_macro']:continue
        assert np.isclose(float(r['reference_mean']),float(row(ds,'XBone-Net')[k+'_mean']))
        assert np.isclose(float(r['baseline_mean']),float(row(ds,r['baseline'])[k+'_mean']))
        checks.append((ds,r['baseline'],k))
p=ROOT/'results/summary/ablation/statistics/paired_bootstrap_results.csv'
for r in csv.DictReader(p.open(encoding='utf-8-sig')):
    name='Phase 2 only' if 'phase2_only' in r['variant_experiment'] else 'Concatenation'
    assert np.isclose(float(r['variant_mean']),float(row('ctch',name)[r['metric']+'_mean']))
    checks.append(('ctch',name,r['metric']))
(BASE/'evidence/statistics_verification.json').write_text(json.dumps({'matched_statistic_rows':len(checks),'checks':checks},indent=2),encoding='utf-8')
print('Verified',len(checks),'paired-statistics rows; prepared tables and templates.')

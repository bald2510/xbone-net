"""Render manuscript tables from the audited, aligned predictions."""
from pathlib import Path
import json

folder = Path(__file__).resolve().parent
audit = json.loads((folder/'evidence/audit.json').read_text(encoding='utf-8'))
data = {(r['dataset'],r['model']):r for r in audit['classification']}
author_results = json.loads((folder/'evidence/author_supplied_results.json').read_text(encoding='utf-8'))
for row in author_results['classification']:
    data[row['dataset'],row['model']] = row
baseline = ['ResNet-50','DenseNet-121','CLIP','MedCLIP','PubMedCLIP','BiomedCLIP']
models = baseline + ['Image-only','XBone-Net']

def cells(ds, model, ranking):
    if (ds,model) not in data:
        return ['---']*3
    row = data[ds,model]
    out=[]
    for j in range(3):
        order=sorted([m for m in ranking if (ds,m) in data], key=lambda m:data[ds,m]['mean'][j],reverse=True)
        value=f"{100*row['mean'][j]:.2f}\\pm{100*row['sd'][j]:.2f}"
        if model==order[0]: value=r'\mathbf{'+value+'}'
        elif model==order[1]: value=r'\underline{'+value+'}'
        out.append('$'+value+'$')
    return out

lines=[r'\begin{table*}[t]',
r'\caption{Classification performance (\%, mean $\pm$ SD across three seeds). Best results are bold; second-best results are underlined. The proposed configurations are separated from the six baselines and their names are bold.}',
r'\label{tab:classification}',r'\centering\footnotesize',r'\setlength{\tabcolsep}{6pt}',
r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcccccc@{}}',r'\toprule',
r'Method & \multicolumn{3}{c}{CTCH (22 classes; $n=669$)} & \multicolumn{3}{c}{BTXRD (10 classes; $n=750$)} \\',
r'\cmidrule(lr){2-4}\cmidrule(l){5-7}',
r' & Accuracy & Balanced acc. & Macro-F1 & Accuracy & Balanced acc. & Macro-F1 \\',r'\midrule']
for model in models:
    if model=='Image-only': lines.append(r'\midrule')
    name = {'Image-only':'XBone-Net Image-only (ours)','XBone-Net':'XBone-Net (ours)'}.get(model,model)
    if model in ['Image-only','XBone-Net']: name=r'\textbf{'+name+'}'
    lines.append(name+' & '+' & '.join(cells('ctch',model,models)+cells('btxrd',model,models))+r' \\')
lines += [r'\bottomrule',r'\end{tabular*}',r'\end{table*}']
(folder/'classification_table.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')

models=['XBone-Net','Phase 2 only','Concatenation']
lines=[r'\begin{table}[t]',
r'\caption{CTCH architecture ablations (\%, mean $\pm$ SD across three seeds). Best results are bold; second-best results are underlined. The Image-only control is reported in Table~\ref{tab:classification}.}',
r'\label{tab:ablation}',r'\centering\footnotesize',r'\setlength{\tabcolsep}{3.5pt}',
r'\begin{tabular}{@{}lccc@{}}',r'\toprule',
r'Configuration & Accuracy & Balanced acc. & Macro-F1 \\',r'\midrule']
for model in models:
    name=r'\textbf{XBone-Net (ours)}' if model=='XBone-Net' else model
    lines.append(name+' & '+' & '.join(cells('ctch',model,models))+r' \\')
lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
(folder/'ablation_table.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('Rendered classification and ablation tables from evidence/audit.json.')

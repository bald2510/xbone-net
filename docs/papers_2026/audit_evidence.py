"""Recompute draft tables from aligned stored test predictions; never modify results."""
from pathlib import Path
import csv, hashlib, json
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / 'evidence'
OUT.mkdir(exist_ok=True)
models = {'XBone-Net':'proposed/ours_xbone_net',
          **{n:f'baselines/full_finetuned/fft_{k}' for n,k in [('ResNet-50','resnet50'),('DenseNet-121','densenet'),('CLIP','clip'),('MedCLIP','medclip'),('PubMedCLIP','pubmedclip'),('BiomedCLIP','biomedclip')]},
          'Phase 2 only':'ablation_study/architecture/phase/phase2_only',
          'Concatenation':'ablation_study/architecture/fusion/concat',
          'Image only':'ablation_study/modality/image_only'}
rows, sources, mismatches = [], [], []
for ds in ['ctch','btxrd']:
    reference = None
    for name, rel in models.items():
        for seed in [42,123,456]:
            run = ROOT/'results'/ds/rel/f'seed_{seed}'
            path = run/'analysis/features'/f'{ds}_test.npz'
            if not path.exists():
                print('MISSING',ds,name,seed)
                continue
            a = np.load(path,allow_pickle=True)
            ids = a['image_id'].astype(str)
            ix = np.argsort(ids)
            labels, pred = a['labels'][ix], a['predictions'][ix]
            if reference is None: reference = (ids[ix],labels)
            assert np.array_equal(reference[0],ids[ix]), (ds,name,'IDs')
            assert np.array_equal(reference[1],labels), (ds,name,'labels')
            assert np.array_equal(a['predictions'],a['probabilities'].argmax(1)), (name,'argmax')
            vals = dict(accuracy=accuracy_score(labels,pred),balanced_accuracy=balanced_accuracy_score(labels,pred),f1_macro=f1_score(labels,pred,average='macro',zero_division=0))
            rows.append(dict(dataset=ds,model=name,seed=seed,n=len(labels),**vals))
            metrics = json.loads((run/'metrics.json').read_text(encoding='utf-8'))['metrics']
            for key,val in vals.items():
                if key in metrics and not np.isclose(val,metrics[key],atol=1e-8):
                    mismatches.append(dict(dataset=ds,model=name,seed=seed,metric=key,cache=val,metrics_json=metrics[key]))
            sources.append(dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),provenance=str(a['provenance_json']) if 'provenance_json' in a else None))
agg=[]
for ds in ['ctch','btxrd']:
    for name in models:
        rr=[r for r in rows if r['dataset']==ds and r['model']==name]
        if len(rr)!=3: continue
        obj=dict(dataset=ds,model=name,n=rr[0]['n'],n_seeds=3)
        for key in ['accuracy','balanced_accuracy','f1_macro']:
            obj[key+'_mean']=float(np.mean([r[key] for r in rr]))
            obj[key+'_sd']=float(np.std([r[key] for r in rr],ddof=1))
        agg.append(obj)
for file,rr in [('classification_per_seed.csv',rows),('classification_aggregate.csv',agg),('source_conflicts.csv',mismatches)]:
    with (OUT/file).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
(OUT/'sources.json').write_text(json.dumps(sources,ensure_ascii=False,indent=2),encoding='utf-8')
for r in agg:
    print(r['dataset'],r['model'],' | '.join(f'{r[k+"_mean"]:.4f} +/- {r[k+"_sd"]:.4f}' for k in ['accuracy','balanced_accuracy','f1_macro']))
print('Conflicting metric cells:',len(mismatches))

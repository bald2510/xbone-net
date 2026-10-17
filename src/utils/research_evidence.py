"""Content-addressed research evidence, independent of mutable metrics.json."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import numpy as np


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),default=str).encode()).hexdigest()


def tensor_state_digest(state, keys):
    import torch
    h=hashlib.sha256()
    for key in sorted(keys):
        tensor=state[key].detach().cpu().contiguous()
        h.update(key.encode())
        h.update(str((tuple(tensor.shape),tensor.dtype)).encode())
        h.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def save_research_checkpoint(model, path):
    """Save trained state/buffers and authenticate all omitted frozen tensors."""
    import torch
    state=model.state_dict()
    learned={k for k,p in model.named_parameters() if p.requires_grad}
    learned.update(k for k,_ in model.named_buffers())
    learned.update(k for k in state if not k.startswith('backbone.'))
    selected={k:v.detach().cpu() for k,v in state.items() if k in learned}
    omitted=sorted(set(state)-set(selected))
    torch.save({'format':'xbone_research_v1','model_state_dict':selected,
                'omitted_keys':omitted,'omitted_sha256':tensor_state_digest(state,omitted)},path)


def load_research_checkpoint(model, payload):
    """Reject a different base model or incomplete compact key coverage."""
    if payload.get('format')!='xbone_research_v1':
        raise ValueError('Not a versioned research checkpoint.')
    current=model.state_dict()
    learned=payload['model_state_dict']
    omitted=payload['omitted_keys']
    if set(learned)&set(omitted) or set(learned)|set(omitted)!=set(current):
        raise ValueError('Compact checkpoint key coverage differs from model.')
    if tensor_state_digest(current,omitted)!=payload['omitted_sha256']:
        raise ValueError('Frozen base mismatch in compact checkpoint.')
    result=model.load_state_dict(learned,strict=False)
    if set(result.missing_keys)!=set(omitted) or result.unexpected_keys:
        raise ValueError('Unexpected compact checkpoint load result.')
    return result


def write_json(path, value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,default=str)+'\n',encoding='utf-8')
    os.replace(tmp,path)


def source_inventory(root):
    root=Path(root)
    paths=[root/'train.py',root/'evaluate.py']
    for folder, suffix in [('src','*.py'),('configs','*.yaml'),('tools/ieee_access','*.py')]:
        paths.extend((root/folder).rglob(suffix))
    return {p.relative_to(root).as_posix():sha256_file(p) for p in sorted(set(paths)) if p.is_file()}


def assert_source_snapshot(root, snapshot):
    if source_inventory(root) != snapshot:
        raise RuntimeError('Source/config snapshot changed. Stop and version the protocol before continuing.')


def validate_predictions(ids, labels, probabilities, class_names):
    ids=np.asarray(ids).astype(str)
    labels=np.asarray(labels)
    p=np.asarray(probabilities)
    if p.ndim!=2 or p.shape!=(len(ids),len(class_names)) or labels.shape!=(len(ids),):
        raise ValueError('Prediction shape/class order mismatch.')
    if not len(ids) or len(set(ids))!=len(ids):
        raise ValueError('Prediction IDs must be nonempty and unique.')
    if not np.isfinite(p).all() or (p<0).any() or (p>1).any() or not np.allclose(p.sum(1),1,atol=1e-5,rtol=0):
        raise ValueError('Invalid probability matrix.')
    if not np.issubdtype(labels.dtype,np.integer) or (labels<0).any() or (labels>=p.shape[1]).any():
        raise ValueError('Invalid class labels.')


def save_prediction_bundle(directory, split, ids, labels, probabilities, class_names, provenance, logits=None):
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    validate_predictions(ids,labels,probabilities,class_names)
    path=directory/f'{split}_predictions.npz'
    if path.exists():
        raise FileExistsError(f'Refusing to replace prediction evidence: {path}')
    arrays=dict(image_id=np.asarray(ids).astype(str), labels=np.asarray(labels,dtype=np.int64),
                probabilities=np.asarray(probabilities), class_names=np.asarray(class_names).astype(str),
                provenance_json=np.array(json.dumps(provenance,sort_keys=True)))
    if logits is not None:
        arrays['logits']=np.asarray(logits)
    tmp=path.with_suffix('.tmp.npz')
    np.savez_compressed(tmp,**arrays)
    os.replace(tmp,path)
    from sklearn.metrics import accuracy_score, recall_score, f1_score
    from src.utils.metrics import multiclass_calibration_metrics
    pred=np.asarray(probabilities).argmax(1)
    metrics={'accuracy':float(accuracy_score(labels,pred)),
             'balanced_accuracy':float(recall_score(labels,pred,labels=np.arange(len(class_names)),average='macro',zero_division=0)),
             'f1_macro':float(f1_score(labels,pred,labels=np.arange(len(class_names)),average='macro',zero_division=0)),
             **multiclass_calibration_metrics(probabilities,labels)}
    write_json(directory/f'{split}_metrics.json',{'metrics':metrics,'prediction_sha256':sha256_file(path),
        'n':len(ids),'split':split,'class_names':class_names,'provenance':provenance})
    return metrics

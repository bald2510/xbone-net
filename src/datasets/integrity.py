"""Strict manifest checks for versioned research protocols."""
import pandas as pd


def merge_manifests(split: pd.DataFrame, labels: pd.DataFrame, *, strict: bool):
    if strict:
        for name, frame in [('split', split), ('labels', labels)]:
            if 'image_id' not in frame or frame.image_id.isna().any():
                raise ValueError(f'{name} manifest requires non-null image_id.')
            if frame.image_id.duplicated().any():
                raise ValueError(f'{name} manifest contains duplicate image_id.')
        if set(split.image_id) != set(labels.image_id):
            raise ValueError('Split and label ID sets differ; refusing a lossy inner join.')
        if split['split'].isna().any() or not set(split['split']).issubset({'train','validate','test'}):
            raise ValueError('Manifest splits must be train, validate, or test.')
        if 'patient_key' in split:
            if split.patient_key.isna().any():
                raise ValueError('Null patient key in strict manifest.')
            if split.groupby('patient_key')['split'].nunique().gt(1).any():
                raise ValueError('A patient key occurs in multiple splits.')
    return pd.merge(split, labels, on='image_id', how='inner', validate='one_to_one')

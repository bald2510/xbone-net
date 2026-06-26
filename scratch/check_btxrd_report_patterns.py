import os
import pandas as pd
from collections import defaultdict

split_df = pd.read_csv("data/BTXRD/btxrd-split.csv")
labels_df = pd.read_csv("data/BTXRD/btxrd-labels.csv")
df = pd.merge(split_df, labels_df, on=["image_id"], how="inner")

test_df = df[df['split'] == 'test'].reset_index(drop=True)

class_reports = defaultdict(list)
clinical_dir = "data/BTXRD/reports/clinical"

for idx, row in test_df.iterrows():
    img_id = row['image_id']
    class_id = int(row['class_id'])
    basename = os.path.splitext(img_id)[0]
    clinical_file = os.path.join(clinical_dir, f"{basename}.txt")
    if os.path.exists(clinical_file):
        with open(clinical_file, "r", encoding="utf-8") as f:
            text = f.read().strip()
            class_reports[class_id].append(text)

# Let's print 3 reports for each class
classes_names = [
    'normal', 'osteochondroma', 'osteosarcoma', 'multiple osteochondromas',
    'simple bone cyst', 'other bt', 'giant cell tumor', 'synovial osteochondroma',
    'other mt', 'osteofibroma'
]

for class_id in range(10):
    name = classes_names[class_id]
    reports = class_reports[class_id]
    print(f"\n=========================================")
    print(f"Class {class_id}: {name} (Total: {len(reports)} reports)")
    print(f"=========================================")
    for i, r in enumerate(reports[:3]):
        print(f"Report {i}: {r}")

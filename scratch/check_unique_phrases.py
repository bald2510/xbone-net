import os
import pandas as pd
from collections import defaultdict

split_df = pd.read_csv("data/BTXRD/btxrd-split.csv")
labels_df = pd.read_csv("data/BTXRD/btxrd-labels.csv")
df = pd.merge(split_df, labels_df, on=["image_id"], how="inner")

clinical_dir = "data/BTXRD/reports/clinical"

phrases = {
    0: "no clear bone abnormalities or fracture identified",  # normal
    1: "bony projection",  # osteochondroma
    2: "destructive lesion with codman triangle",  # osteosarcoma
    3: "multiple palpable exostoses",  # multiple osteochondromas
    4: "cystic lesion",  # simple bone cyst
    5: "without significant pain",  # other bt
    6: "joint effusion",  # giant cell tumor
    7: "mechanical locking",  # synovial osteochondroma
    8: "constitutional symptoms",  # other mt
    9: "cortical lesion"  # osteofibroma
}

# Let's count how many test samples for each class contain their class phrase
class_counts = defaultdict(int)
class_correct_phrases = defaultdict(int)

for idx, row in df.iterrows():
    img_id = row['image_id']
    class_id = int(row['class_id'])
    split = row['split']
    
    basename = os.path.splitext(img_id)[0]
    clinical_file = os.path.join(clinical_dir, f"{basename}.txt")
    
    if os.path.exists(clinical_file):
        with open(clinical_file, "r", encoding="utf-8") as f:
            text = f.read().lower()
            
        class_counts[class_id] += 1
        
        # Check if the correct phrase is in the text
        phrase = phrases[class_id]
        if phrase in text:
            class_correct_phrases[class_id] += 1
        else:
            # Let's print which sample of class_id does not have the phrase
            if class_id == 8:
                print(f"Class 8 sample {basename} does not have phrase '{phrase}':")
                print("  Text:", text)

print("\n--- Phrase Matching Statistics ---")
for class_id in sorted(phrases.keys()):
    total = class_counts[class_id]
    correct = class_correct_phrases[class_id]
    pct = (correct / total) * 100 if total > 0 else 0.0
    print(f"Class {class_id} ({phrases[class_id]:35s}): {correct}/{total} ({pct:.1f}%)")

import os
import pandas as pd

classes = [
    'normal',
    'osteochondroma',
    'osteosarcoma',
    'multiple osteochondromas',
    'simple bone cyst',
    'other bt',
    'giant cell tumor',
    'synovial osteochondroma',
    'other mt',
    'osteofibroma'
]

split_df = pd.read_csv("data/BTXRD/btxrd-split.csv")
test_images = split_df[split_df['split'] == 'test']['image_id'].tolist()
print(f"Total test images: {len(test_images)}")

clinical_dir = "data/BTXRD/reports/clinical"
xray_dir = "data/BTXRD/reports/xray"

clinical_matches = 0
xray_matches = 0

for img_id in test_images:
    basename = os.path.splitext(img_id)[0]
    
    # Clinical report
    clinical_file = os.path.join(clinical_dir, f"{basename}.txt")
    if os.path.exists(clinical_file):
        with open(clinical_file, "r", encoding="utf-8") as f:
            text = f.read().lower()
            found = [c for c in classes if c in text]
            if found:
                clinical_matches += 1
                # Print first few matches for inspection
                if clinical_matches <= 5:
                    print(f"[Clinical] Found in {basename}: {found}")
                    print("  Text:", text)
                    
    # Xray report
    xray_file = os.path.join(xray_dir, f"{basename}.txt")
    if os.path.exists(xray_file):
        with open(xray_file, "r", encoding="utf-8") as f:
            text = f.read().lower()
            found = [c for c in classes if c in text]
            if found:
                xray_matches += 1
                if xray_matches <= 5:
                    print(f"[Xray] Found in {basename}: {found}")
                    print("  Text:", text)

print(f"\nClinical reports containing class name: {clinical_matches} / {len(test_images)}")
print(f"Xray reports containing class name: {xray_matches} / {len(test_images)}")

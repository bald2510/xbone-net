import pandas as pd

split_df = pd.read_csv("data/BTXRD/btxrd-split.csv")
labels_df = pd.read_csv("data/BTXRD/btxrd-labels.csv")

print("Split dataframe columns:", split_df.columns)
print("Labels dataframe columns:", labels_df.columns)

print("\nValue counts of split:")
print(split_df['split'].value_counts())

train_images = set(split_df[split_df['split'] == 'train']['image_id'])
val_images = set(split_df[split_df['split'] == 'validate']['image_id'])
test_images = set(split_df[split_df['split'] == 'test']['image_id'])

print(f"\nTrain images: {len(train_images)}")
print(f"Val images: {len(val_images)}")
print(f"Test images: {len(test_images)}")

# Check for overlap
print("Intersection Train & Val:", len(train_images.intersection(val_images)))
print("Intersection Train & Test:", len(train_images.intersection(test_images)))
print("Intersection Val & Test:", len(val_images.intersection(test_images)))

# Check if there is a patient/subject ID column in the labels or splits to check for subject-level leakage
if 'subject_id' in split_df.columns:
    train_subs = set(split_df[split_df['split'] == 'train']['subject_id'])
    val_subs = set(split_df[split_df['split'] == 'validate']['subject_id'])
    test_subs = set(split_df[split_df['split'] == 'test']['subject_id'])
    print("\nIntersection Train & Test subjects:", len(train_subs.intersection(test_subs)))
else:
    print("\nNo subject_id column found in split_df")

if 'patient_id' in split_df.columns:
    train_pts = set(split_df[split_df['split'] == 'train']['patient_id'])
    val_pts = set(split_df[split_df['split'] == 'validate']['patient_id'])
    test_pts = set(split_df[split_df['split'] == 'test']['patient_id'])
    print("Intersection Train & Test patients:", len(train_pts.intersection(test_pts)))
else:
    print("No patient_id column found in split_df")

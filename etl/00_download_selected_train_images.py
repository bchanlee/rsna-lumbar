import pandas as pd

# Step 1: Load train.csv to get unique (study_id, series_id) pairs
train_csv_path = "../data/train_label_coordinates.csv"  # adjust path if needed
df = pd.read_csv(train_csv_path)
pairs = df[['study_id', 'series_id']].drop_duplicates().astype(str)

series_folder = []
study_id = []
series_id = []
for _, row in pairs.iterrows():
    study_id.append(row['study_id'])
    series_id.append(row['series_id'])
    series_folder.append(f"/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_images/{row['study_id']}/{row['series_id']}")
    
    # if len(study_id) == 293:
    #     break

# Create DataFrame
df = pd.DataFrame({
    'series_folder': series_folder,
    'study_id': study_id,
    'series_id': series_id
})

coords_df = pd.read_csv("../data/train_label_coordinates.csv")
coords_df = coords_df.drop_duplicates(subset=['study_id', 'series_id']) # unintentional duplicates
coords_df = coords_df[["study_id", "series_id", "instance_number"]]
coords_df["study_id"] = coords_df["study_id"].astype(str)
coords_df["series_id"] = coords_df["series_id"].astype(str)
df = df.merge(coords_df, on=["study_id", "series_id"], how="inner")
df.to_csv("../data/series_folder.csv", index=False)
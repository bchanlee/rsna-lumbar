import glob
import os
import pandas as pd

def create_training_df(image_dir, condition):
	images = glob.glob(os.path.join(image_dir, "*.png"))
	df = pd.DataFrame({"filepath": images})
	df["filepath"] = df.filepath.map(os.path.basename)
	df["study_id"] = df.filepath.apply(lambda x: x.split("_")[0]).astype("int")
	return df

	
coords_df = pd.read_csv("../data/train_label_coordinates.csv")
filepath_df = create_training_df("../data/train_pngs_3ch/spinal/", "spinal")
fold_df = pd.read_csv("../data/folds_cv5.csv")

# Pivot the coordinates dataframe
pivot_df = coords_df.pivot_table(
    index='study_id',
    columns='level',
    values=['x', 'y']
)

# Flatten MultiIndex columns (e.g., ('x', 'L1/L2') -> 'l1_l2_x')
pivot_df.columns = [f"{lvl.lower().replace('/', '_')}_{coord}" for coord, lvl in pivot_df.columns]
pivot_df = pivot_df.reset_index()


# Merge with filepath dataframe
final_df = filepath_df.merge(pivot_df, on='study_id', how='left')
final_df = final_df.merge(fold_df, on="study_id", how="inner")
final_df = final_df.dropna()

print(final_df.shape)

final_df.to_csv("../data/train_sagittal_t2_coords_regression_kfold.csv", index=False)
# generate 3-channel images (slice-1, slice, slice+1) of the wanted slice from grey-scale png
import cv2
import numpy as np
import os
import pandas as pd

from tqdm import tqdm


def get_condition(string):
	string = string.lower()
	for condition in ["spinal", "foraminal", "subarticular"]:
		if condition in string:
			return condition

coords_df = pd.read_csv("../data/train_label_coordinates.csv")
coords_df = coords_df.drop_duplicates(subset=['study_id', 'series_id']) # unintentional duplicates

image_dir = "../data/train_pngs/"
save_dir = "../data/train_pngs_3ch/"

failed = []
for row in tqdm(coords_df.itertuples(), total=len(coords_df)):
	this_instance = row.instance_number
	ch2 = this_instance - 1
	ch1 = max(0, ch2 - 1)
	ch3 = ch2 + 1
	instances_3ch = [ch1, ch2, ch3]
	images = np.stack([
		cv2.imread(os.path.join(image_dir, str(row.study_id), str(row.series_id), f"IM{_:06d}.png"), 0)
		for _ in instances_3ch
	], axis=-1)
	condition = get_condition(row.condition)
	savefile = os.path.join(save_dir, condition, f"{row.study_id}_{row.series_id}_{this_instance}.png")
	os.makedirs(os.path.dirname(savefile), exist_ok=True)
	try:
		status = cv2.imwrite(savefile, images)
	except Exception as e:
		print(row.series_id, e)
		failed.append(row)
		
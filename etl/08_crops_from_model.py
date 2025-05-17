import albumentations as A
import cv2
import glob
import numpy as np
import os
import pandas as pd
import sys
sys.path.insert(0, "../../rsna-lumbar")
import torch

from collections import defaultdict
from importlib import import_module
from tqdm import tqdm

def load_model_fold_dict(checkpoint_dict, cfg):
    model_dict = {}
    cfg.pretrained = False
    for fold, checkpoint_path in checkpoint_dict.items():
        print(f"Loading weights from {checkpoint_path} ...")
        wts = torch.load(checkpoint_path)["state_dict"]
        wts = {k.replace("model.", ""): v for k, v in wts.items()}
        model = import_module(f"models.{cfg.model}").Net(cfg)
        model.load_state_dict(wts)
        model = model.eval()
        model_dict[fold] = model
    return model_dict

def crop_square_around_center(img, xc, yc, size_factor=0.15):
	h, w = size_factor * img.shape[0], size_factor * img.shape[1]
	x1, y1 = xc - w / 2, yc - h / 2
	x2, y2 = x1 + w, y1 + h
	x1, y1, x2, y2 = [int(_) for _ in [x1, y1, x2, y2]]
	return img[y1:y2, x1:x2]

def save_list_of_images(image_list, study_id, series_id, laterality, save_dir):
    levels = ["L1_L2", "L2_L3", "L3_L4", "L4_L5", "L5_S1"]
    filenames = [f"{study_id}_{series_id}_{laterality}_{lvl}.png" if laterality != "" else f"{study_id}_{series_id}_{lvl}.png" for lvl in levels]
    assert len(image_list) == len(levels)
    os.makedirs(save_dir, exist_ok=True)
    for fname, img in zip(filenames, image_list):
        if not isinstance(img, np.ndarray):
            continue
        status = cv2.imwrite(os.path.join(save_dir, fname), img)

cfg_file = "cfg_spinal_reg"
cfg = import_module(f"configs.{cfg_file}").cfg
# do rest of the folds too
checkpoint_dict = {
    "0": "../experiments/cfg_spinal_reg/b52ada95/fold0/checkpoints/epoch=018-val_metric=0.0395.ckpt"
}
canal_localisation_model = {
    "cfg": cfg,
    "models": load_model_fold_dict(checkpoint_dict, cfg)
}

folds_df = pd.read_csv("../data/folds_cv5.csv")
study_id_fold_dict = {int(row.study_id): row.fold for row in folds_df.itertuples()}
description_df = pd.read_csv("../data/train_series_descriptions.csv")
study_series_id_description_dict = {f"{row.study_id}-{row.series_id}": row.series_description for row in description_df.itertuples()}
image_dir = "../data/train_pngs_3ch/"
save_dir = "../data/train_generated_crops/"
levels = ["L1", "L2", "L3", "L4", "L5", "S1"]
levels_dict = {ii: lvl for ii, lvl in enumerate(levels)}
xy_partition = 5
condition = "spinal"

index = 0
for study_id, fold in tqdm(study_id_fold_dict.items(), total=len(study_id_fold_dict)):
    if fold != 0:
        continue
    series = glob.glob(os.path.join(image_dir, condition, f"{study_id}_*"))
    series_path_dict = defaultdict(list)
    for each_series in series:
        series_id = os.path.basename(each_series).split("_")[1]
        series_path_dict[study_series_id_description_dict[f"{study_id}-{series_id}"]].append(each_series)
    
    SAG_T2_AVAILABLE = len(series_path_dict["Sagittal T2/STIR"]) > 0
    if SAG_T2_AVAILABLE:
        sag_t2_file = series_path_dict["Sagittal T2/STIR"][0]
        sag_t2 = cv2.imread(sag_t2_file, cfg.cv2_load_flag)
        sag_t2_np = canal_localisation_model["cfg"].val_transforms(image=sag_t2, keypoints=[])["image"]
        sag_t2_np = sag_t2_np.transpose(2, 0, 1)
        sag_t2_torch = torch.from_numpy(sag_t2_np).unsqueeze(0)
        
        with torch.inference_mode():
            canal_out = canal_localisation_model["models"][str(fold)]({"x": sag_t2_torch})["logits"].sigmoid().cpu().numpy()[0]
            h, w = sag_t2.shape[0], sag_t2.shape[1]
            canal_out[:xy_partition] = canal_out[:xy_partition] * w
            canal_out[xy_partition:] = canal_out[xy_partition:] * h
            # print(study_id, canal_out)

        canal_out = canal_out.astype("int")
        canal_out = np.stack([canal_out[:xy_partition], canal_out[xy_partition:]], axis=0)
        cropped_canals = []
        for level in range(len(levels) - 1):
            cropped_canal = crop_square_around_center(sag_t2, xc=canal_out[0, level], yc=canal_out[1, level], size_factor=0.15)
            cropped_canals.append(cropped_canal)

        save_list_of_images(cropped_canals, study_id, series_id, laterality="", save_dir=os.path.join(save_dir, condition))


    # if index == 1:
    #     break
    # index += 1
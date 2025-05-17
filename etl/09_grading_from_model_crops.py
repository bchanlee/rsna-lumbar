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

cfg_file = "cfg_spinal_cls"
cfg = import_module(f"configs.{cfg_file}").cfg
# # do rest of the folds too
checkpoint_dict = {
    "0": "../experiments/cfg_spinal_cls/797f3f95/fold0/checkpoints/epoch=007-val_metric=0.2523.ckpt",
}
canal_cls_model = {
    "cfg": cfg,
    "models": load_model_fold_dict(checkpoint_dict, cfg)
}

folds_df = pd.read_csv("../data/folds_cv5.csv")
study_id_fold_dict = {int(row.study_id): row.fold for row in folds_df.itertuples()}
description_df = pd.read_csv("../data/train_series_descriptions.csv")
study_series_id_description_dict = {f"{row.study_id}-{row.series_id}": row.series_description for row in description_df.itertuples()}
image_dir = "../data/train_generated_crops/"
condition = "spinal"
num_classes = 3
preds = []
study_ids = []
levels = []

index = 0
for study_id, fold in tqdm(study_id_fold_dict.items(), total=len(study_id_fold_dict)):
    if fold != 0:
        continue
    series = glob.glob(os.path.join(image_dir, condition, f"{study_id}_*"))
    series_path_dict = defaultdict(list)
    for each_series in series:
        series_id = os.path.basename(each_series).split("_")[1]
        level = os.path.basename(each_series).replace(".png", "")[-5:]
        series_path_dict[study_series_id_description_dict[f"{study_id}-{series_id}"]].append(each_series)
        study_ids.append(study_id)
        levels.append(level)

    SAG_T2_AVAILABLE = len(series_path_dict["Sagittal T2/STIR"]) > 0
    if SAG_T2_AVAILABLE:
        for series_ind in range(len(series_path_dict["Sagittal T2/STIR"])):
            sag_t2_file = series_path_dict["Sagittal T2/STIR"][series_ind]
            sag_t2 = cv2.imread(sag_t2_file, cfg.cv2_load_flag)
            sag_t2_np = canal_cls_model["cfg"].val_transforms(image=sag_t2)["image"]
            sag_t2_np = sag_t2_np.transpose(2, 0, 1)
            sag_t2_torch = torch.from_numpy(sag_t2_np).unsqueeze(0)

            with torch.inference_mode():
                pred = canal_cls_model["models"][str(fold)]({"x": sag_t2_torch})["logits"].softmax(dim=1).cpu().numpy()[0]
                pred = np.eye(num_classes, dtype=int)[np.argmax(pred)]
                preds.append(pred)

    # if index == 1:
    #     break
    # index += 1


assert len(preds) == len(study_ids) == len(levels)

# Transpose to get class-wise data
transposed = np.array(preds).T
# Assign to variables
normal_mild, moderate, severe = transposed

pred_df = pd.DataFrame({
    "study_id": study_ids,
    "level": levels,
    "normal_mild_pred_gen": normal_mild,
    "moderate_pred_gen": moderate,
    "severe_pred_gen": severe
})

pred_df.to_csv("../data/generated_crops_preds.csv", index=False)
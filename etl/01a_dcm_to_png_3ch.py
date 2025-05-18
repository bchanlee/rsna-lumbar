import albumentations as A
import cv2
import glob
import numpy as np
import os
import pandas as pd
import pydicom

from tqdm import tqdm


def get_image_plane(vals):
    vals = [round(v) for v in vals]
    plane = np.cross(vals[:3], vals[3:6])
    plane = [abs(x) for x in plane]
    return np.argmax(plane) # 0- sagittal, 1- coronal, 2- axial


def convert_to_8bit(x):
    lower, upper = np.percentile(x, (1, 99))
    x = np.clip(x, lower, upper)
    x = x - np.min(x)
    x = x / np.max(x) 
    return (x * 255).astype("uint8")


def load_dicom_stack(dicom_folder, sort_mode="instance"):
    assert sort_mode in ["instance", "position"]
    dicom_files = glob.glob(os.path.join(dicom_folder, "*.dcm"))
    dicoms = [pydicom.dcmread(f) for f in dicom_files]
    plane = get_image_plane(dicoms[0].ImageOrientationPatient)
    instances = [int(d.InstanceNumber) for d in dicoms]
    positions = [float(d.ImagePositionPatient[plane]) for d in dicoms]
    idx1, idx2 = np.argsort(instances), np.argsort(positions)
    # sometimes sorting based on position is just the reverse
    mismatch = int(not np.array_equal(idx1, idx2) and not np.array_equal(idx1, idx2[::-1]))
    array_shapes = np.vstack([d.pixel_array.shape for d in dicoms])
    h, w = array_shapes[:, 0].max(), array_shapes[:, 1].max()
    padder = A.PadIfNeeded(min_height=h, min_width=w, p=1, border_mode=cv2.BORDER_CONSTANT, value=0)
    array = [padder(image=d.pixel_array.astype("float32"))["image"] for d in dicoms]
    array = np.stack(array)
    if mismatch == 1:
        print(f"{dicom_folder}: mismatch between image instance and position")
    if sort_mode == "instance":
        array = array[idx1]
        return array, np.sort(instances), mismatch
    elif sort_mode == "position":
        array = array[idx2]
        return array, np.arange(len(positions)), mismatch


DATA_DIR = "/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification"
# SAVE_DIR = os.path.join("/kaggle/working/", "train_pngs")
save_dir = os.path.join("/kaggle/working/", "train_pngs_3ch", "spinal")

series_df = pd.read_csv("/kaggle/input/rsna-lumbar-data/series_folder.csv")

failed, mismatches = [], []
total_series = len(series_df.series_id.unique())
# for each_series, tmp_series_df in tqdm(series_df.groupby("series_id"), total=total_series):
for each_series in tqdm(series_df["series_id"], total=len(series_df)):
    try:
        tmp_series_df = series_df[series_df['series_id'] == each_series]
        study_id = tmp_series_df.study_id.iloc[0]
        # tmp_save_dir = os.path.join(SAVE_DIR, str(study_id), str(each_series))
        # os.makedirs(tmp_save_dir, exist_ok=True)
        stack, instances, mismatch = load_dicom_stack(tmp_series_df.series_folder.iloc[0], sort_mode="instance")
        this_instance = tmp_series_df.instance_number.iloc[0]
        ch2 = this_instance
        ch1 = max(0, ch2-1)
        ch3 = min(len(instances), ch2+1)
        if mismatch:
            mismatches.append(each_series)
        stack = convert_to_8bit(stack)
        arr1 = None
        arr2 = None
        arr3 = None
        for each_slice, each_instance in zip(stack, instances):
            if each_instance == ch1:
                arr1 = each_slice
            if each_instance == ch2:
                arr2 = each_slice
            if each_instance == ch3:
                arr3 = each_slice
                # sts = cv2.imwrite(os.path.join(tmp_save_dir, f"IM{each_instance:06d}.png"), each_slice)
        images = np.stack([arr1, arr2, arr3], axis=-1)
        savefile = os.path.join(save_dir, f"{study_id}_{each_series}_{this_instance}.png")
        os.makedirs(os.path.dirname(savefile), exist_ok=True)
        sts = cv2.imwrite(savefile, images)
    except Exception as e:
        print(f"FAILED {each_series}: {e}")
        failed.append(each_series)
    
    # break
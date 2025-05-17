# Overview
Mini project on Spinal Canal Stenosis diagnosis from MRI images (n=300 pts, 5000 dicom img)

Assume sagittal T2 images with slice number given for good stenosis grading

Steps
1. Keypoint localisation of spinal stenosis for each level in lumbar spine
	- targets: x, y coordinates for L1-S1
2. Classifcation of stenosis grading based on crops from keypoint localisation
	- targets: normal/mild, moderate, severe
	- model is trained on crops from ground truth coordinates
	

## Performance
### Step 1

| Fold | Best Validation MAESigmoid |
|------|------------------------|
| 0    | 0.0395                 |

### Step 2
CV 0.2740

| Fold | Best Validation SampleWeightedLogLoss |
|------|------------------------|
| 0    | 0.2523                 |
| 1    | 0.3081                 |
| 2    | 0.2850                 |
| 3    | 0.2897                 |
| 4    | 0.2345                 |


Fold 0
Using ground truth crops
- True Normal/Mild: 255 / 260
- True Moderate: 0 / 15
- True Severe: 8 / 11
- True Total: 263 / 286

Using generated crops
- True Normal/Mild: 234 / 260
- True Moderate: 1 / 15
- True Severe: 7 / 11
- True Total: 242 / 286


# TODO
- [x] get general overview which files to use
- [x] get dataset (not full) and setup dir, environment.sh
- [x] not zip fully so convert dicom to png using one of the etl files
- [x] train spinal canal stensosis grading
- [x] look at output from best checkpoints
- [x] regression
- [x] inference to generate coordinates by scaling to original image then cropping (etl/000d_use_models_to_generate_crops.py)

- [ ] slice localisation (can just use half) - assume slice given
- [ ] update conda/pip environment list
- [ ] cnn-transformer for localisation
- [ ] think about using augs (train_generated_crops_with_augs_dist_coord_proba)
- [ ] think about using wandb instead of neptune logger - disabled for now
- [ ] double CV? why need inner folds (hyperparam tuning but where in the code)
	
	
# Past Issues
- downloading from kaggle output doesnt download all but using kaggle competitions downloads all
	- 500 file limit
	- make datasets
	- no metadata - may cause slice issues (assume instance number is always in the middle)
- can’t download select nested files using kaggle competitions CLI
- downloading from saved version kaggle stops almost 1 GB
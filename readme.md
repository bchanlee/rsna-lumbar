# Overview
Targets
Foraminal narrowing (on either the left or right foramen at a specified level).
Subarticular stenosis (on either the left or right side at a specified level).
Canal stenosis (only at a specified level).

Just do Canal stenosis grading (from crops given ground truth coordinates), Sagittal T2 / STIR slices, fewer 
	full dataset: 1975 sagittal T2 series (assuming all unique, and 1-to-1 with study id)
	test dataset: no coordinates given so need localisation step (different model e.g. CNN-transformer)

`data`
spinal stenosis sagittal T2 5000 dicom images, 300 pts

## Performance
### cls: spinal stenosis grading (normal/mild, mod, severe)
fold best_val_metric
0	0.2523
1	0.3081
2	0.285
3	0.2897
4	0.2345
CV 0.274
Fold 0
True Normal / Mild: 255/260
True Moderate: 0/15
True Severe: 8/11
True Total: 263/286

### regression: spinal stenosis coordinates (l1 to s1, x, y)
fold best_val_metric
0 0.0521
1 0.0616
2 0.0487
3 not done
4 not done
CV 0.054

# TODO
- [x] get general overview which files to use
- [x] get dataset (not full) and setup dir, environment.sh
- [x] not zip fully so convert dicom to png using one of the etl files
- [x] train spinal canal stensosis grading
- [x] look at output from best checkpoints
- [x] regression

- [ ] inference to generate coordinates by scaling to original image then cropping (etl/000d_use_models_to_generate_crops.py)
- [ ] slice localisation (can just use half)
- [ ] update conda/pip environment list
- [ ] cnn-transfomrer for localisation
- [ ] think about using augs (train_generated_crops_with_augs_dist_coord_proba)
- [ ] think about using wandb instead of neptune logger - disabled for now
	
	
# Past Issues
- downloading from kaggle output doesnt download all but using kaggle competitions downloads all
	- 500 file limit
	- make datasets
	- no metadata - may cause slice issues (assume instance number is always in the middle)
- can’t download select nested files using kaggle competitions CLI
- downloading from saved version kaggle stops almost 1 GB
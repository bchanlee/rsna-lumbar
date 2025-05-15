# Overview
Targets
Foraminal narrowing (on either the left or right foramen at a specified level).
Subarticular stenosis (on either the left or right side at a specified level).
Canal stenosis (only at a specified level).

Just do Canal stenosis grading (from crops given ground truth coordinates), Sagittal T2 / STIR slices, fewer 
	full dataset: 1975 sagittal T2 series (assuming all unique, and 1-to-1 with study id)
	test dataset: no coordinates given so need localisation step (different model e.g. CNN-transformer)

classification of spinal canal stenosis severity (normal_mild, moderate, severe) from ground truth stenosis
> cfg0_gt_spinal_crops.py

## ~data
spinal stenosis sagittal T2 5000 dicom images, 300 pts

# TODO
[x] get general overview which files to use
[x] get dataset (not full) and setup dir, environment.sh
	[x] not zip fully so convert dicom to png using one of the etl files
	
	
# ISSUES
- downloading from kaggle output doesnt download all but using kaggle competitions downloads all
	- 500 file limit
	- make datasets
	- no metadata - may cause slice issues
- can’t download select nested files using kaggle competitions CLI
- downloading from saved version kaggle stops almost 1 GB



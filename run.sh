#!/bin/bash

python train.py cfg_spinal_cls --accelerator mps --devices 1 --precision 32 --strategy auto --fold 0
python train.py cfg_spinal_cls --accelerator mps --devices 1 --precision 32 --strategy auto --fold 1
python train.py cfg_spinal_cls --accelerator mps --devices 1 --precision 32 --strategy auto --fold 2
python train.py cfg_spinal_cls --accelerator mps --devices 1 --precision 32 --strategy auto --fold 3
python train.py cfg_spinal_cls --accelerator mps --devices 1 --precision 32 --strategy auto --fold 4
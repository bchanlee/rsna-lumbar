#!/bin/bash

# Train spinal stenosis cls
# python train.py cfg_spinal_cls --accelerator mps --devices 1 --precision 32 --strategy auto --fold 0
# python train.py cfg_spinal_cls --accelerator mps --devices 1 --precision 32 --strategy auto --fold 1
# python train.py cfg_spinal_cls --accelerator mps --devices 1 --precision 32 --strategy auto --fold 2
# python train.py cfg_spinal_cls --accelerator mps --devices 1 --precision 32 --strategy auto --fold 3
# python train.py cfg_spinal_cls --accelerator mps --devices 1 --precision 32 --strategy auto --fold 4

# Eval spinal stenosis cls
python eval.py cfg_spinal_cls experiments/cfg_spinal_cls/797f3f95/fold0/checkpoints/epoch=007-val_metric=0.2523.ckpt

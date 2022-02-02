#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -p compsci-gpu
#SBATCH -o outputs/test.out 
cd CURRENT_DIRECTORY
python3 test.py \
    --dataset_root PATH_TO_SINTEL_CHAIRS \
    --flowEst_root PATH_TO_ESTIMATED_FLOW \
    --is_train 2\
    --experiment_name MONet_ft3d\
    --load_weights MONet_ft3d/weights-ft3d.hdf5 
#    --save_preds

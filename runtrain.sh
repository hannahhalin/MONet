#!/bin/bash
#SBATCH --gres=gpu:1  
#SBATCH -p compsci-gpu
#SBATCH -o outputs/train_MONet.out
cd CURRENT_DIRECTORY
python3 train.py \
    --dataset_root PATH_TO_FlyingThings3D \
    --flowEst_root PATH_TO_ESTIMATED_FLOW  \
    --is_train 1\
    --experiment_name YOUR_EXPERIMENT_NAME\
    --load_weights MONet_ft3d/weights-ft3d.hdf5 

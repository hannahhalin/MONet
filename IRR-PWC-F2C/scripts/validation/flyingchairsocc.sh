#!/bin/bash
#SBATCH -t 200:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=compsci-gpu
#SBATCH -o eval-%j.out

# experiments and datasets meta
EXPERIMENTS_HOME=?/evaluation

# datasets
FLYINGCHAIRS_OCC_HOME=?/FlyingChairsOcc/data

# model and checkpoint
MODEL=IRR_PWC_FED_OEE
CHECKPOINT="pretrained_model/checkpoint_IRR-PWC-F2C-flyingchairsocc.ckpt"
EVAL_LOSS=MultiScaleEPE_PWC_Bi_Occ_upsample

SIZE_OF_BATCH=1

# validate clean configuration
SAVE_PATH="$EXPERIMENTS_HOME/$MODEL-flyingchairsocc"
python3 ../../main.py \
--batch_size=$SIZE_OF_BATCH \
--batch_size_val=$SIZE_OF_BATCH \
--checkpoint=$CHECKPOINT \
--evaluation=True \
--save_result_occ=False \
--save_result_img=False \
--model=$MODEL \
--num_workers=4 \
--save=$SAVE_PATH \
--validation_dataset=FlyingChairsOccValid \
--validation_dataset_photometric_augmentations=False \
--validation_dataset_root=$FLYINGCHAIRS_OCC_HOME \
--validation_key=occ_F1 \
--validation_loss=$EVAL_LOSS \
--validation_key_minimize=False
# IRR-PWC-F2C on Occlusion Estimation
Here is the code for our Fine-to-Coarse version of IRR-PWC model

## Requirements
```Shell
conda create -n IRR-PWC-F2C
conda activate IRR-PWC-F2C
conda install pytorch==1.5.0 torchvision==0.6.0 -c pytorch
tqdm (conda install -c conda-forge tqdm==4.40.0)
Install other missing packages. Most of them can be installed by pip install <package_name>.
```


## Required Data
FlyingChairsOcc and FlyingThings3D subset are used for training the model, and Sintel is used for evaluation.

Please refer [here](https://github.com/visinf/irr#getting-started) to download the corresponding datasets.

## Training
Please see examples in the training scripts 
  - Training on the FlyingChairsOcc: scripts/training/train_flyingchairsocc.sh
  - Training on the FlyingThings3D: scripts/training/train_flyingthings3D.sh

## Evaluation
Please see examples in the evaluation scripts
  - Evaluate on Sintel Clean training set: scripts/validation/sintel_clean.sh
  - Evaluate on Sintel Final training set: scripts/validation/sintel_final.sh
  - Evaluate on FlyingChairsOcc validation set: scripts/validation/flyingchairsocc.sh

## Acknowledment & Reference
  - IRR-PWC-F2C is adapted from(https://github.com/visinf/irr) by visinf.


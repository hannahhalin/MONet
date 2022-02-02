# Motion Boundary and Occlusion Nework (MONet)
This repository contains the code for our paper:<br/>
[Joint Detection of Motion Boundaries and Occlusions](https://arxiv.org/pdf/2111.01261.pdf)<br/>
BMVC 2021 <br/>
Hannah Halin Kim, Shuzhi Yu, and Carlo Tomasi<br/>
<img src=predictions/alley_1.gif>
<img src=predictions/bandage_1.gif>
<img src=predictions/temple_3.gif>

## Citations
Please cite our paper if you find our code or paper useful.
```
@inproceedings{kim2021joint,
      title={Joint Detection of Motion Boundaries and Occlusions}, 
      author={Hannah Halin Kim and Shuzhi Yu and Carlo Tomasi},
      booktitle=BMVC,
      year={2021},
}
```


## Requirements
The code has been developed with Tensorflow 1.15.0 and Keras 2.3.1.
```Shell
conda create -n MONet tensorflow-gpu==1.15.0 keras==2.3.1  
conda activate MONet 
pip3 install numpy==1.19.5 scikit-image opencv-python 'h5py<3.0.0' matplotlib
```


## Required Data
To train MONet, you will need to download:
  - [FlyingThings3D](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)

To evaluate MONet, you will need to download: 
  - [Sintel](http://sintel.is.tue.mpg.de/) 
  - [FlyingChairsOcc](https://github.com/visinf/irr/tree/master/flyingchairsocc)

You also need to save their corresponding estimated flow maps in both temporal directions to use as an input to MONet.


## Training
You can train a model using `train.py` (`runtrain.sh`). 
Training logs will be written to the `experiments/` which can be visualized using tensorboard.
Pretrained model trained on FlyingThings3D dataset is in `experiments/MONet_ft3d/weights-ft3d.hdf5`.
```Shell
python3 train.py \
    --dataset_root PATH_TO_FlyingThings3D \
    --flowEst_root PATH_TO_ESTIMATED_FLOW \
    --is_train 1\
    --experiment_root YOUR_EXPERIMENT_NAME\
    --load_weights MONet_ft3d/weights-ft3d.hdf5 
```

Please set your own experiment name (`YOUR_EXPERIMENT_NAME`) and path to your training dataset directory in your local system (`PATH_TO_FlyingThings3D`, `PATH_TO_ESTIMATED_FLOW`).


## Evaluation
You can evaluate a trained model using `test.py` (`runtest.sh`) 
Pretrained model is in `experiments/MONet_ft3d/weights-ft3d.hdf5`.
```Shell
python3 test.py \
    --dataset_root PATH_TO_SINTEL_CHAIRS \
    --flowEst_root PATH_TO_ESTIMATED_FLOW \
    --is_train 2\
    --experiment_root MONet_ft3d\
    --load_weights MONet_ft3d/weights-ft3d.hdf5
```
Please set path to your testing dataset directory in your local system (`PATH_TO_SINTEL_CHAIRS`, `PATH_TO_ESTIMATED_FLOW`). 
You can save the motion boundary and occlusion predictions to `predictions/` by adding `--save_preds`. 

Occlusion predictions are evaluated in `test.py`.
Motion boundary predictions can be evaluated using this [code](https://github.com/lmb-freiburg/mbeval) by lmb-freiburg.


## IRR-PWC-F2C
Please see `IRR-PWC-F2C/` for the fine-to-coarse version of [irr](https://github.com/visinf/irr) by visinf (IRR-PWC).


## Acknowledment & Reference
  - [tfoptflow](https://github.com/philferriere/tfoptflow) by philferriere.
  - [netdef_models](https://github.com/lmb-freiburg/netdef_models) by lmb-freiburg.
  - [irr](https://github.com/visinf/irr) by visinf.
  - [UnFlow](https://github.com/simonmeister/UnFlow) by simonmeister.



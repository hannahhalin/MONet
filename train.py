import os
import numpy as np
import tensorflow.compat.v1 as tf
from keras import optimizers
from keras.callbacks import ReduceLROnPlateau, TensorBoard, LambdaCallback
from argument_parser import myParser
from data_generators import FT3D_Dataset
from loss import *
from monet import monet

def main():

    args = myParser()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, 
                                            log_device_placement=False))

    # loading network 
    net = monet(args.sizeV, args.sizeH)

    if args.load_weights is not None:
        print ('Loading Network Weights: '+args.load_weights)
        net.load_weights('experiments/'+args.load_weights, by_name=True)
        
    mylosses, myweights = getMyJointLosses()

    if args.optimizer_type == 'adam':
        myO = optimizers.Adam(lr=args.learning_rate)
    elif args.optimizer_type == 'sgd':
        myO = optimizers.SGD(lr=args.learning_rate, decay=0.0001, momentum=0.9)

    net.compile(loss=mylosses, loss_weights=myweights, optimizer=myO) 

    # data generators
    train_generator = FT3D_Dataset(args)
    val_args = args
    val_args.is_train = 0 # Validation Set
    validation_generator = FT3D_Dataset(val_args)
        
    # checkpoints
    filepath = os.path.join('experiments',args.experiment_name, 
                            "weights-{:02d}.hdf5")
    checkpoint = LambdaCallback(on_epoch_end=lambda epoch,
                                logs:net.save_weights(filepath.format(epoch)))

    # learning rate schedule
    lrate = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=10)
    
    # tensorboard
    tensorboard = TensorBoard(log_dir=os.path.join("experiments", 
                                                   args.experiment_name, 
                                                   "logs"))
    callbacks_list = [checkpoint, lrate, tensorboard]

    # train
    net.fit_generator(train_generator,
                      validation_data=validation_generator, 
                      validation_steps=validation_generator.batch_count,
                      steps_per_epoch=train_generator.batch_count, 
                      epochs=args.num_epochs, workers=12, verbose=2,
                      callbacks=callbacks_list, initial_epoch=args.init_epoch)

if __name__ == '__main__':
    main()

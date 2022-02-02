import os
import sys
import h5py 
import numpy as np
import tensorflow.compat.v1 as tf
from loss import *
from data_generators import pyramid_inputs, load_sintel, load_sintel_fids
from data_generators import load_fcocc, load_fcocc_fids
from monet import monet
import matplotlib.pyplot as plt
from optflow import *
from scipy.io import savemat
from keras import optimizers
from argument_parser import myParser
    
def save_predictions(args, preds, preds2, fid, mbI=1, occI=1):
    preds_occ = np.zeros((args.sizeV_orig, args.sizeH, 1))
    preds_mb = np.zeros((args.sizeV_orig, args.sizeH, 1))
    preds_occb = np.zeros((args.sizeV_orig, args.sizeH, 1))
    preds_mbb = np.zeros((args.sizeV_orig, args.sizeH, 1))

    mbInitI = 0 # 0-5
    mbbInitI = 12 # 12-17
    occInitI = 6 # 6-11
    occbInitI = 18 # 18-23
    
    if args.save_preds:    
        if not os.path.exists('predictions/'+args.experiment_name+
                              fid[:fid[1:].find('/')+1]):
            os.mkdirs('predictions/'+args.experiment_name+
                      fid[:fid[1:].find('/')+1])
            
    # mb predictions
    if mbI == 1:
        preds_mb[:args.sizeV_orig//2+1,:,0] \
            = np.squeeze(preds[mbInitI+5])[:args.sizeV_orig//2+1,:]
        preds_mb[-args.sizeV_orig//2:,:,0] \
            = np.squeeze(preds2[mbInitI+5])[-args.sizeV_orig//2:,:]
        preds_mbb[:args.sizeV_orig//2+1,:,0] \
            = np.squeeze(preds[mbbInitI+5])[:args.sizeV_orig//2+1,:]
        preds_mbb[-args.sizeV_orig//2:,:,0] \
            = np.squeeze(preds2[mbbInitI+5])[-args.sizeV_orig//2:,:]

        # save images
        if args.save_preds:
            plt.imsave('predictions/'+args.experiment_name+fid+'_mb.png',
                       np.squeeze(preds_mb), vmin=0.0, vmax=1.0)
            np.save('predictions/'+args.experiment_name+fid+'_mb.npy', 
                    np.squeeze(preds_mb))
            plt.imsave('predictions/'+args.experiment_name+fid+'_mbb.png',
                       np.squeeze(preds_mbb), vmin=0.0, vmax=1.0)
            np.save('predictions/'+args.experiment_name+fid+'_mbb.npy', 
                    np.squeeze(preds_mbb))
                    
    # occ predictions
    if occI == 1:
        preds_occ[:args.sizeV_orig//2+1,:,0] \
            = np.squeeze(preds[occInitI+5])[:args.sizeV_orig//2+1,:]
        preds_occ[-args.sizeV_orig//2:,:,0] \
            = np.squeeze(preds2[occInitI+5])[-args.sizeV_orig//2:,:]
        preds_occb[:args.sizeV_orig//2+1,:,0] \
            = np.squeeze(preds[occbInitI+5])[:args.sizeV_orig//2+1,:]
        preds_occb[-args.sizeV_orig//2:,:,0] \
            = np.squeeze(preds2[occbInitI+5])[-args.sizeV_orig//2:,:]
        
        # save images
        if args.save_preds:
            plt.imsave('predictions/'+args.experiment_name+fid+'_occ.png',
                       np.squeeze(preds_occ), vmin=0.0, vmax=1.0)
            np.save('predictions/'+args.experiment_name+fid+'_occ.npy', 
                    np.squeeze(preds_occ))
            plt.imsave('predictions/'+args.experiment_name+fid+'_occb.png',
                       np.squeeze(preds_occb), vmin=0.0, vmax=1.0)
            np.save('predictions/'+args.experiment_name+fid+'_occb.npy', 
                    np.squeeze(preds_occb))

    return preds_mb, preds_mbb, preds_occ, preds_occb

def prepare_data(args, img1s, img2s, sps, sps2, flowEsts, bflowEsts):

    img1 = img1s[:,:args.sizeV,:,:]
    img2 = img2s[:,:args.sizeV,:,:]
    sp = sps[:,:args.sizeV,:,:]
    sp2 = sps2[:,:args.sizeV,:,:]
    flowEst = flowEsts[:,:args.sizeV,:,:]
    bflowEst = bflowEsts[:,:args.sizeV,:,:]
        
    img1b = img1s[:,-args.sizeV:,:,:]
    img2b = img2s[:,-args.sizeV:,:,:]
    sp_2= sps[:,-args.sizeV:,:,:]
    sp2_2 = sps2[:,-args.sizeV:,:,:]
    flowEstb = flowEsts[:,-args.sizeV:,:,:]
    bflowEstb = bflowEsts[:,-args.sizeV:,:,:]
            
    # multi-scale inputs
    img1s, sps, flowEsts = pyramid_inputs(img1, flowEst)
    img2s, sps2, bflowEsts = pyramid_inputs(img2, bflowEst)
    img1s_2, sps_2, flowEstsb = pyramid_inputs(img1b, flowEstb)
    img2s_2, sps2_2, bflowEstsb = pyramid_inputs(img2b, bflowEstb)

    X = [img1]+ img1s + [img2] + img2s + [sp] + sps + [sp2] + sps2 + \
        [flowEst] + flowEsts + [bflowEst] + bflowEsts
    X2 = [img1b]+ img1s_2 + [img2b] + img2s_2 + [sp_2] + sps_2 + \
         [sp2_2] + sps2_2 + [flowEstb] + flowEstsb + [bflowEstb] + bflowEstsb
    
    return X, X2    

def main():

    args = myParser()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device
    gpu_options = tf.GPUOptions(allow_growth=True, 
                                per_process_gpu_memory_fraction=0.3)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, 
                      allow_soft_placement=True, log_device_placement=True))

    if 'Sintel' in args.dataset_root:
        print('Evaluating on MPI-Sintel dataset...')
        fids = load_sintel_fids(args.dataset_root)
        args.sizeV_orig = 436
        args.sizeV = 416
        args.sizeH = 1024
    elif 'FlyingChairsOcc' in args.dataset_root:
        print('Evaluating on FlyingChairsOcc dataset...')
        fids = load_fcocc_fids(args.dataset_root)
        args.sizeV_orig = 384
        args.sizeV = 256
        args.sizeH = 512
    else:
        print('Evaluating on a custom dataset...')
        sys.exit('Need to setup dataloader for the custom dataset '
                 '(data_generators.py).')

    # loading network
    net = monet(args.sizeV, args.sizeH)

    if args.load_weights is not None:
        print ('Loading Network Weights: '+args.load_weights)
        net.load_weights('experiments/'+args.load_weights, by_name=True)
        
    # losses
    mylosses, myweights = getMyJointLosses()

    if args.optimizer_type == 'adam':
        myO = optimizers.Adam(lr=args.learning_rate)
    elif args.optimizer_type == 'sgd':
        myO = optimizers.SGD(lr=args.learning_rate, decay=0.0001, momentum=0.9)

    net.compile(loss=mylosses, loss_weights=myweights, optimizer=myO)

    if 'Sintel' in args.dataset_root:
        if 'final' in args.dataset_root:
            finalI = True
            args.experiment_name = args.experiment_name+'_final'
        elif 'clean' in args.dataset_root:
            finalI = False
            args.experiment_name = args.experiment_name+'_clean'
    elif 'FlyingChairsOcc' in args.dataset_root:
        args.experiment_name = args.experiment_name+'_fcocc'

    if args.save_preds:
        print ('Saving images in '+'predictions/'+args.experiment_name)
        if not os.path.exists('predictions/'+args.experiment_name):
            os.mkdir('predictions/'+args.experiment_name)
                     
    beta=1
    beta2 = beta ** 2
    eps=1e-8       
    precision = []
    recall = []
    f1 = []
    for i in range(len(fids)):
        fid = fids[i]

        # load inputs
        if 'Sintel' in args.dataset_root:
            img1s, img2s, sps, sps2, occ1, flowEst, bflowEst \
                = load_sintel(args.dataset_root, args.flowEst_root, 
                              fid=fids[i], finalI=finalI)
        elif 'FlyingChairsOcc' in args.dataset_root:
            img1s, img2s, sps, sps2, occ1, occ2, flowEst, bflowEst \
                = load_fcocc(args.dataset_root, args.flowEst_root, fid=fids[i])
        
        X, X2 = prepare_data(args, img1s, img2s, sps, sps2, flowEst, bflowEst)

        # predict
        preds = net.predict(X)
        preds2 = net.predict(X2)

        # save
        preds_mb, preds_mbb, preds_occ, preds_occb \
            = save_predictions(args, preds, preds2, fids[i])

        # occlusion performance 
        preds_occ=np.squeeze(preds_occ>0.5).astype(float)
        occ1=np.squeeze(occ1>0.5).astype(float)
        true_positive=np.sum(preds_occ*occ1)        
        precision.append(true_positive/(np.sum(preds_occ)+eps))
        recall.append(true_positive/(np.sum(occ1)+eps))
        f1.append(precision[-1]*recall[-1]/(precision[-1]*beta2+recall[-1] \
                  + eps) * (1 + beta2))
        
    print(' ')
    print('Occlusion Estimation:')
    print('   Average precision: '+str(np.mean(precision)))
    print('   Average recall: '+str(np.mean(recall)))
    print('   Average F1: '+str(np.mean(f1)))
    print(' ')
    print('Motion Boundary Estimation:')
    print('   Check README file for instructions.')

if __name__ == '__main__':
    main()

import numpy as np
import random
import matplotlib.image as mpimg
from skimage.segmentation import slic, find_boundaries
from skimage.transform import pyramid_gaussian
from optflow import flow_read
from keras.utils import Sequence
import os
from glob import glob


def load_ft3d_fids(root, is_train):
    """ 
    Gathers a list of filenames of FlyingThings3D dataset
    
    Args:
        root: path to the FlyingThings3D dataset
        is_train: 1 - training set, 0 - validation set
       
    Returns:
        fids: numpy string arrays listing path to each image pair
    """

    cameraType = 'right'
    if is_train==1:
        folderN = 'train'
    else:
        folderN = 'val'

    dirlist = sorted(os.listdir(root+'/'+folderN+'/motion_boundaries/'+
                                cameraType+'/into_past/'))
    dirlist_flow = sorted(os.listdir(root+'/'+folderN+'/flow/'+cameraType+
                                     '/into_future/'))
    dirlist_flowb = sorted(os.listdir(root+'/'+folderN+'/flow/'+cameraType+
                                      '/into_past/'))
    
    if is_train ==0:
        dirlist = dirlist[::40]

    fcount = 0
    fids = {}
    for thisFile in dirlist:
        if thisFile[:-4]+'.flo' in dirlist_flow:
            if str(int(thisFile[:-4])+1).zfill(7)+'.flo' in dirlist_flowb:
                fids[fcount] = str(thisFile[:-4])
                fcount = fcount + 1
            else:
                print('    Backward Flow label does not exist :'+thisFile)
        else:
            print('    Flow label does not exist :'+thisFile)

    print(   'Loading '+str(fcount)+' image pairs for FlyingThigs3D '+folderN+' dataset.')
    return fids


def load_sintel_fids(root):
    """ 
    Gathers a list of filenames of MPI-Sintel dataset
    
    Args:
        root: path to the dataset used for testing.
       
    Returns:
        fids: numpy string arrays listing path to each image pair
    """
    
    rootlist = sorted(os.listdir(root))
    dirlist = [ item for item in rootlist if os.path.isdir(os.path.join(
                                                           root, item)) ]
    fcount = 0
    fids = {}

    for folderN in dirlist:
        rootf = root + '/' + str(folderN)
        rootflist = sorted(os.listdir(rootf))
        fileslist = [f for f in rootflist]
        for thisFile in fileslist[0:-1]:
            if '._' not in thisFile:
                fids[fcount] =  '/' +str(folderN)+'/'+str(thisFile[:-4])
                fcount = fcount + 1
    
    print('Loading '+str(fcount)+' image pairs for MPI-Sintel dataset.')
    return fids


def load_fcocc_fids(root):
    """ 
    Gathers a list of filenames of FlyingChairsOcc dataset
    
    Args:
        root: path to the dataset used for testing.
       
    Returns:
        fids: numpy string arrays listing path to each image pair
    """
    
    pids = sorted(glob(os.path.join(root, "*_img1.png")))

    # Remove invalid validation indices
    VALIDATE_INDICES = [
        5, 17, 42, 45, 58, 62, 96, 111, 117, 120, 121, 131, 132,
        152, 160, 248, 263, 264, 291, 293, 295, 299, 316, 320, 336,
        337, 343, 358, 399, 401, 429, 438, 468, 476, 494, 509, 528,
        531, 572, 581, 583, 588, 593, 681, 688, 696, 714, 767, 786,
        810, 825, 836, 841, 883, 917, 937, 942, 970, 974, 980, 1016,
        1043, 1064, 1118, 1121, 1133, 1153, 1155, 1158, 1159, 1173,
        1187, 1219, 1237, 1238, 1259, 1266, 1278, 1296, 1354, 1378,
        1387, 1494, 1508, 1518, 1574, 1601, 1614, 1668, 1673, 1699,
        1712, 1714, 1737, 1841, 1872, 1879, 1901, 1921, 1934, 1961,
        1967, 1978, 2018, 2030, 2039, 2043, 2061, 2113, 2204, 2216,
        2236, 2250, 2274, 2292, 2310, 2342, 2359, 2374, 2382, 2399,
        2415, 2419, 2483, 2502, 2504, 2576, 2589, 2590, 2622, 2624,
        2636, 2651, 2655, 2658, 2659, 2664, 2672, 2706, 2707, 2709,
        2725, 2732, 2761, 2827, 2864, 2866, 2905, 2922, 2929, 2966,
        2972, 2993, 3010, 3025, 3031, 3040, 3041, 3070, 3113, 3124,
        3129, 3137, 3141, 3157, 3183, 3206, 3219, 3247, 3253, 3272,
        3276, 3321, 3328, 3333, 3338, 3341, 3346, 3351, 3396, 3419,
        3430, 3433, 3448, 3455, 3463, 3503, 3526, 3529, 3537, 3555,
        3577, 3584, 3591, 3594, 3597, 3603, 3613, 3615, 3670, 3676,
        3678, 3697, 3723, 3728, 3734, 3745, 3750, 3752, 3779, 3782,
        3813, 3817, 3819, 3854, 3885, 3944, 3947, 3970, 3985, 4011,
        4022, 4071, 4075, 4132, 4158, 4167, 4190, 4194, 4207, 4246,
        4249, 4298, 4307, 4317, 4318, 4319, 4320, 4382, 4399, 4401,
        4407, 4416, 4423, 4484, 4491, 4493, 4517, 4525, 4538, 4578,
        4606, 4609, 4620, 4623, 4637, 4646, 4662, 4668, 4716, 4739,
        4747, 4770, 4774, 4776, 4785, 4800, 4845, 4863, 4891, 4904,
        4922, 4925, 4956, 4963, 4964, 4994, 5011, 5019, 5036, 5038,
        5041, 5055, 5118, 5122, 5130, 5162, 5164, 5178, 5196, 5227,
        5266, 5270, 5273, 5279, 5299, 5310, 5314, 5363, 5375, 5384,
        5393, 5414, 5417, 5433, 5448, 5494, 5505, 5509, 5525, 5566,
        5581, 5602, 5609, 5620, 5653, 5670, 5678, 5690, 5700, 5703,
        5724, 5752, 5765, 5803, 5811, 5860, 5881, 5895, 5912, 5915,
        5940, 5952, 5966, 5977, 5988, 6007, 6037, 6061, 6069, 6080,
        6111, 6127, 6146, 6161, 6166, 6168, 6178, 6182, 6190, 6220,
        6235, 6253, 6270, 6343, 6372, 6379, 6410, 6411, 6442, 6453,
        6481, 6498, 6500, 6509, 6532, 6541, 6543, 6560, 6576, 6580,
        6594, 6595, 6609, 6625, 6629, 6644, 6658, 6673, 6680, 6698,
        6699, 6702, 6705, 6741, 6759, 6785, 6792, 6794, 6809, 6810,
        6830, 6838, 6869, 6871, 6889, 6925, 6995, 7003, 7026, 7029,
        7080, 7082, 7097, 7102, 7116, 7165, 7200, 7232, 7271, 7282,
        7324, 7333, 7335, 7372, 7387, 7407, 7472, 7474, 7482, 7489,
        7499, 7516, 7533, 7536, 7566, 7620, 7654, 7691, 7704, 7722,
        7746, 7750, 7773, 7806, 7821, 7827, 7851, 7873, 7880, 7884,
        7904, 7912, 7948, 7964, 7965, 7984, 7989, 7992, 8035, 8050,
        8074, 8091, 8094, 8113, 8116, 8151, 8159, 8171, 8179, 8194,
        8195, 8239, 8263, 8290, 8295, 8312, 8367, 8374, 8387, 8407,
        8437, 8439, 8518, 8556, 8588, 8597, 8601, 8651, 8657, 8723,
        8759, 8763, 8785, 8802, 8813, 8826, 8854, 8856, 8866, 8918,
        8922, 8923, 8932, 8958, 8967, 9003, 9018, 9078, 9095, 9104,
        9112, 9129, 9147, 9170, 9171, 9197, 9200, 9249, 9253, 9270,
        9282, 9288, 9295, 9321, 9323, 9324, 9347, 9399, 9403, 9417,
        9426, 9427, 9439, 9468, 9486, 9496, 9511, 9516, 9518, 9529,
        9557, 9563, 9564, 9584, 9586, 9591, 9599, 9600, 9601, 9632,
        9654, 9667, 9678, 9696, 9716, 9723, 9740, 9820, 9824, 9825,
        9828, 9863, 9866, 9868, 9889, 9929, 9938, 9953, 9967, 10019,
        10020, 10025, 10059, 10111, 10118, 10125, 10174, 10194,
        10201, 10202, 10220, 10221, 10226, 10242, 10250, 10276,
        10295, 10302, 10305, 10327, 10351, 10360, 10369, 10393,
        10407, 10438, 10455, 10463, 10465, 10470, 10478, 10503,
        10508, 10509, 10809, 11080, 11331, 11607, 11610, 11864,
        12390, 12393, 12396, 12399, 12671, 12921, 12930, 13178,
        13453, 13717, 14499, 14517, 14775, 15297, 15556, 15834,
        15839, 16126, 16127, 16386, 16633, 16644, 16651, 17166,
        17169, 17958, 17959, 17962, 18224, 21176, 21180, 21190,
        21802, 21803, 21806, 22584, 22857, 22858, 22866]
    fcount = 0
    fids = {}
    for i in VALIDATE_INDICES: 
      pid = pids[i]
      fids[fcount] = pid[pid.rfind('/'):-9]
      fcount = fcount + 1

    print('Loading '+str(len(fids))+' image pairs for FlyingChairsOcc '
          'dataset.')
    return fids



def load_sintel_mb(gtmb_root, fid):
    """
    Retrieve specified motion boundary label map from MPI-Sintel dataset

    Args:
      dataset_root: path to the saved true motion boudnary map
      fid: file name of the MPI-Sintel data to load

    Returns:
      mb: true motion boundary map
    """
    mb = mpimg.imread(gtmb_root+fid+'.png')
    mb = np.round(np.clip(mb, 0, 1))
    mb = np.expand_dims(mb, axis=0)
    return mb
    
def load_sintel_occ(fid, dataset_root):
    """
    Retrieve specified occlusion label map from MPI-Sintel dataset

    Args:
      dataset_root: path to MPI-Sintel dataset
      fid: file name of the MPI-Sintel data to load

    Returns:
      occ: true occlusion map
    """
    
    occ_path = dataset_root[:dataset_root.rfind('training/')+8]+ \
               '/occlusions_rev/'+fid+'.png'
    occ = np.squeeze(np.mean(mpimg.imread(occ_path)>0, 2))
    occ = np.round(np.clip(occ, 0, 1))
    occ = np.expand_dims(occ, axis=0)
    return occ
     
def load_sintel_flow(dataset_root, fid):
    """
    Retrieve specified flow label map from MPI-Sintel dataset

    Args:
      dataset_root: path to MPI-Sintel dataset
      fid: file name of the MPI-Sintel data to load

    Returns:
      flow: true flow map
    """

    flow = flow_read(dataset_root[:dataset_root.rfind('training/')+8]+ \
                     '/flow/'+fid+'.flo')
    flow = np.expand_dims(flow, axis=0)
    return flow


def load_sintel(dataset_root, flowEst_root, fid='', finalI=False, spN=2000):
    """
    Retrieve specified input samples from MPI-Sintel dataset for MONet

    Args:
      dataset_root: path to MPI-Sintel dataset
      dataset_root: path to estimated flow maps for MPI-Sintel dataset
      fid: file name of the MPI-Sintel data to load
      finalI: True - final version, False - clean version of MPI-Sintel
      spN: number of superpixels to segment in each image

    Returns:
      img1: image frame 1
      img2: image frame 2
      sp1: superpixel boundary map of img1
      sp2: superpixel boundary map of img2
      flowEst: forward flow estimation map
      bflowEst: backward flow estimation map
    """
    # image pairs
    img1 = np.expand_dims(mpimg.imread(dataset_root+fid+'.png'), axis=0)
    img2 = np.expand_dims(mpimg.imread(dataset_root+fid[:-4]+"%04d" % \
                                       (int(fid[-4:]) + 1)+'.png'), axis=0)
    
    # superpixel boundaries
    sp1 = np.expand_dims(find_boundaries(slic(img1, n_segments=spN, 
                         multichannel=True)).astype(np.uint8), axis=3)
    sp2 = np.expand_dims(find_boundaries(slic(img2, n_segments=spN, 
                         multichannel=True)).astype(np.uint8), axis=3 )
    # true occlusion
    occ = load_sintel_occ(fid, dataset_root)

    # flow estimation
    flowEst, bflowEst = loadFlowEst(flowEst_root, fid, finalI)
    flowEst = np.expand_dims(flowEst, axis=0)
    bflowEst = np.expand_dims(bflowEst, axis=0)

    return img1, img2, sp1, sp2, occ, flowEst, bflowEst


def loadFlowEst(flowEst_root, fid='', finalI=False):
    """
    Retrieve specified flow estimation map for MPI-Sintel dataset

    Args:
      flowEst_root: path to forward flow estimation maps for MPI-Sintel dataset
      fid: file name of the MPI-Sintel data to load
      finalI: True - final version, False - clean version of MPI-Sintel

    Returns:
      flowEst: forward flow estimation map
      bflowEst: backward flow estimation map
    """

    if finalI:
        dataTypeStr = 'final'
    else:
        dataTypeStr = 'clean'

    flowEst = flow_read(flowEst_root+dataTypeStr+fid+'.flo')
    bflowEst = flow_read(flowEst_root+dataTypeStr+'_backward'+
                         fid[:-4]+"%04d" % (int(fid[-4:])+1)+'.flo')

    return flowEst, bflowEst


def load_fcocc(dataset_root, flowEst_root, fid='', spN=2000):
    """
    Retrieve specified input samples from FlyingChairsOcc dataset for MONet

    Args:
      dataset_root: path to FlyingChairsOcc dataset
      flowEst_root: path to estimated flow maps for FlyingChairsOcc dataset
      fid: file name of the FlyingChairsOcc data to load
      spN: number of superpixels to segment in each image

    Returns:
      img1: image frame 1
      img2: image frame 2
      sp1: superpixel boundary map of img1
      sp2: superpixel boundary map of img2
      img1: image frame 1
      img2: image frame 2
      flowEst: forward flow estimation map
      bflowEst: backward flow estimation map
    """
    # fid = fid[-14:-9]
    # image pairs
    img1 = np.expand_dims(mpimg.imread(dataset_root+fid+'_img1.png'), axis=0)
    img2 = np.expand_dims(mpimg.imread(dataset_root+fid+'_img2.png'), axis=0)
    occ1 = np.expand_dims(np.mean(mpimg.imread(dataset_root+fid+'_occ1.png')>0,
                                  2, keepdims=True), axis=0)
    occ2 = np.expand_dims(np.mean(mpimg.imread(dataset_root+fid+'_occ2.png')>0,
                                  2), axis=0)
    
    # superpixel boundaries
    sp1 = np.expand_dims(find_boundaries(slic(img1, n_segments=spN, 
                         multichannel=True)).astype(np.uint8), axis=3)
    sp2 = np.expand_dims(find_boundaries(slic(img2, n_segments=spN, 
                         multichannel=True)).astype(np.uint8), axis=3 )

    flowEst = np.expand_dims(flow_read(flowEst_root+fid+'_flow.flo'), axis=0)
    bflowEst = np.expand_dims(flow_read(flowEst_root+fid+'_flow_backward.flo'),
                              axis=0)

    return img1, img2, sp1, sp2, occ1, occ2, flowEst, bflowEst


def pyramid_inputs(img1b, flowEst, spN=2000):
    """
    Retrieve pyramid of the input images, their superpixel boundary maps, 
        and flow estimation maps

    Args:
      img1b: images
      flowEst: flow estimation maps
      spN: number of superpixels to segment in each image

    Returns:
      imgs: pyramid of the input images
      spbs: superpixel boundary maps of the image pyramid
      flowEsts: pyramid of flow estimation map
    """
    p_levels = 5
    sp_rate = 3
    bs = np.shape(img1b)[0]
    for i in range(bs):
        pyramid = tuple(pyramid_gaussian(np.squeeze(img1b[i,:,:,:]), 
                                         downscale=2, max_layer=p_levels, 
                                         multichannel=True))
        if i == 0:
            img1b_a = np.zeros((bs,np.shape(pyramid[1])[0], 
                                np.shape(pyramid[1])[1],3))
            img1b_b = np.zeros((bs,np.shape(pyramid[2])[0], 
                                np.shape(pyramid[2])[1],3))
            img1b_c = np.zeros((bs,np.shape(pyramid[3])[0], 
                                np.shape(pyramid[3])[1],3))
            img1b_d = np.zeros((bs,np.shape(pyramid[4])[0], 
                                np.shape(pyramid[4])[1],3))
            
            spb_a = np.zeros((bs,np.shape(pyramid[1])[0], 
                              np.shape(pyramid[1])[1],1))
            spb_b = np.zeros((bs,np.shape(pyramid[2])[0], 
                              np.shape(pyramid[2])[1],1))
            spb_c = np.zeros((bs,np.shape(pyramid[3])[0], 
                              np.shape(pyramid[3])[1],1))
            spb_d = np.zeros((bs,np.shape(pyramid[4])[0], 
                              np.shape(pyramid[4])[1],1))

        img1b_a[i,] = pyramid[1]
        img1b_b[i,] = pyramid[2]
        img1b_c[i,] = pyramid[3]
        img1b_d[i,] = pyramid[4]
        
        spb_a[i,:,:,0] = find_boundaries(slic(img1b_a[i,:,:,0:3], 
                                         n_segments=spN//(sp_rate**1),
                                         multichannel=True)).astype(np.uint8)
        spb_b[i,:,:,0] = find_boundaries(slic(img1b_b[i,:,:,0:3], 
                                         n_segments=spN//(sp_rate**2),
                                         multichannel=True)).astype(np.uint8)
        spb_c[i,:,:,0] = find_boundaries(slic(img1b_c[i,:,:,0:3], 
                                         n_segments=spN//(sp_rate**3),
                                         multichannel=True)).astype(np.uint8)
        spb_d[i,:,:,0] = find_boundaries(slic(img1b_d[i,:,:,0:3], 
                                         n_segments=spN//(sp_rate**4),
                                         multichannel=True)).astype(np.uint8)
    
    imgs = [img1b_a, img1b_b, img1b_c, img1b_d]
    spbs = [spb_a, spb_b, spb_c, spb_d]
    flowEsts = [flowEst[:, ::2**1, ::2**1, :] / (2**1),
                flowEst[:, ::2**2, ::2**2, :] / (2**2),
                flowEst[:, ::2**3, ::2**3, :] / (2**3),
                flowEst[:, ::2**4, ::2**4, :] / (2**4)]

    return imgs, spbs, flowEsts


def load_ft3d_flow(fid, dataset_root, is_train):
    """
    Retrieve specified flow label map from FlyingThings3D dataset

    Args:
      dataset_root: path to FlyingThings3D dataset
      fid: file name of the FlyingThings3D data to load
      is_train: 1 - training set, 0 - validation set

    Returns:
      gtflow: true forward flow map
      gtflowb: true backward flow map
    """
    if is_train == 1:
        trainIfolder = 'train'
    else:
        trainIfolder = 'val'

    gtflowb = flow_read(dataset_root+'/'+trainIfolder+'/flow/right/into_past/'
                        +(str(int(fid)+1).zfill(7))+'.flo')
    gtflow = flow_read(dataset_root+'/'+trainIfolder+'/flow/right/into_future/'
                       +fid+'.flo')
    return gtflow, gtflowb
  
  
def load_ft3d_occ(fid, dataset_root, is_train):
    """
    Retrieve specified occlusion label map from FlyingThings3D dataset

    Args:
      dataset_root: path to FlyingThings3D dataset
      fid: file name of the FlyingThings3D data to load
      is_train: 1 - training set, 0 - validation set

    Returns:
      occ: true forward occlusion map
      occb: true backward occlusion map
    """
    if is_train ==1:
        trainIfolder = 'train'
    else:
        trainIfolder = 'val'
    occ = mpimg.imread(dataset_root+'/'+trainIfolder+
                       '/flow_occlusions/right/into_future/'+fid+'.png')
    occ = np.round(np.clip(occ, 0, 1))
    occb = mpimg.imread(dataset_root+'/'+trainIfolder+
                        '/flow_occlusions/right/into_past/'+
                        (str(int(fid)+1).zfill(7))+'.png')
    occb = np.round(np.clip(occb, 0, 1))
    return occ, occb
    
def load_ft3ds(dataset_root, flowEst_root, is_train, spN=2000, fids=[], i2=[]):
    """
    Retrieve specified input samples from FlyingThings3D dataset for MONet

    Args:
      dataset_root: path to MPI-Sintel dataset
      flowEst_root: path to estimated flow maps for MPI-Sintel dataset
      is_train: 0 - validation set, 1 - train set 
      spN: number of superpixels to segment in each image
      fids: file name of the MPI-Sintel data to load
      i2: index of samples to retrieve

    Returns:
      img1s: image frames 1
      img2s: image frames 2
      sp1s: superpixel boundary maps of img1
      sp2s: superpixel boundary maps of img2
      gtmbs: true forward motion boundary maps
      gtmbbs: true backward motion boundary maps
      flowEsts: estimated forward flow maps
      bflowEsts: estimated backward flow maps
    """
    if len(i2) == 0:
        iterIs =np.arange(len(fids))
    else:
        iterIs = i2

    b = len(iterIs)
    for i, ival in enumerate(iterIs):
        img1, img2, sp1, sp2, mb, mbb, fEst, bfEst \
            = load_ft3d(fids[ival], dataset_root, flowEst_root, is_train, spN)

        gtflow, gtflowb = load_ft3d_flow(fids[ival], dataset_root, is_train)
        occ, occb = load_ft3d_occ(fids[ival], dataset_root, is_train)
        if i ==0:
            h = np.shape(img1)[0]
            w = np.shape(img1)[1]
            img1s = np.zeros((b, h, w, 3))
            img2s = np.zeros((b, h, w, 3))
            sp1s = np.zeros((b, h, w, 1))
            sp2s = np.zeros((b, h, w, 1))
            gtmbs = np.zeros((b, h,w, 1))
            gtmbbs = np.zeros((b, h,w, 1))
            flowEsts = np.zeros((b, h,w, 2))
            bflowEsts = np.zeros((b, h,w, 2))
            gtflows = np.zeros((b, h, w, 2))
            gtflowbs = np.zeros((b, h, w, 2))
            gtoccs = np.zeros((b, h, w,1))
            gtoccbs = np.zeros((b, h,w ,1))
        
        img1s[i,:,:,:] = img1
        img2s[i,:,:,:] = img2
        sp1s[i,:,:,0] = sp1
        sp2s[i,:,:,0] = sp2
        gtmbs[i,:,:,:] = mb
        gtmbbs[i,:,:,:] = mbb
        flowEsts[i,:,:,:] = fEst
        bflowEsts[i,:,:,:] = bfEst
        gtflows[i,:,:,:] = gtflow
        gtflowbs[i,:,:,:] = gtflowb
        gtoccs[i,:,:,0] = occ
        gtoccbs[i,:,:,0] = occb
    return img1s, img2s, sp1s, sp2s, gtmbs, gtmbbs, flowEsts, bflowEsts, \
                gtflows, gtflowbs, gtoccs, gtoccbs


def load_ft3d(fid, dataset_root, flowEst_root, is_train, spN=2000):
    """
    Retrieve a specified input sample from FlyingThings3D dataset for MONet

    Args:
      fid: file name of the MPI-Sintel data to load
      dataset_root: path to MPI-Sintel dataset
      flowEst_root: path to estimated flow maps for MPI-Sintel dataset
      is_train: 0 - validation set, 1 - train set 
      spN: number of superpixels to segment in each image

    Returns:
      img1: image frame 1
      img2: image frame 2
      sp1: superpixel boundary map of img1
      sp2: superpixel boundary map of img2
      gtmb: true forward motion boundary
      gtmbb: true backward motion boundary 
      flowEst: forward flow estimation map
      bflowEst: backward flow estimation map
    """
    if is_train ==1:
        trainIfolder = 'train'
    else:
        trainIfolder = 'val'
    
    img1 = mpimg.imread(dataset_root+'/'+trainIfolder+'/image_clean/right/'+ 
                        fid+'.png')
    img2 = mpimg.imread(dataset_root+'/'+trainIfolder+'/image_clean/right/'+ 
                        (str(int(fid)+1).zfill(7))+'.png')
    segs = slic(img1, n_segments=spN)
    sp1 = find_boundaries(segs).astype(np.uint8)
    segs2 = slic(img2, n_segments=spN)
    sp2 = find_boundaries(segs2).astype(np.uint8)
    
    gtmb = mpimg.imread(dataset_root+'/'+trainIfolder+ \
                        '/motion_boundaries/right/into_past/'+fid+'.png') 
    gtmb = np.expand_dims(np.round(np.clip(gtmb, 0, 1)), axis=2)
    gtmbb = mpimg.imread(dataset_root+'/'+trainIfolder+ \
                         '/motion_boundaries/right/into_future/'+ \
                         (str(int(fid)+1).zfill(7))+'.png')
    gtmbb = np.expand_dims(np.round(np.clip(gtmbb, 0, 1)), axis=2)

    # Flow estimation input 
    flowEst = flow_read(flowEst_root+'/'+trainIfolder+
                        '/right/into_future/'+fid+'.flo')
    bflowEst = flow_read(flowEst_root+'/'+trainIfolder+
                         '/right/into_past/'+(str(int(fid)+1).zfill(7))+
                         '.flo')
    return img1, img2, sp1, sp2, gtmb, gtmbb, flowEst, bflowEst

        
def crop_ft3ds(sizeV, sizeH, img1, img2, sp1, sp2, Y_mb, Y_mbb, flowEst, 
               bflowEst, Y_flow, Y_flowb, Y_occ, Y_occb, is_train):
    """
    cropping of images and labels to the give size (sizeV, sizeH) for MONet
    
    Args:
       (sizeV, sizeH): dimension to crop
       img1: image frame 1
       img2: image frame 2
       sp1: superpixel boundary map of img1
       sp2: superpixel boundary map of img2
       Y_mb: true forward motion boundary
       Y_mbb: true backward motion boundary 
       flowEst: forward flow estimation map
       bflowEst: backward flow estimation map
       Y_flow: true forward flow map
       Y_flowb: true backward flow map
       Y_occ: true forward occlusion map
       Y_occb: true backward occlusion map

    Returns:
       img1: cropped image frame 1
       img2: cropped image frame 2
       sp1: cropped superpixel boundary map of img1
       sp2: cropped superpixel boundary map of img2
       Y_mb: cropped true forward motion boundary
       Y_mbb: cropped true backward motion boundary 
       flowEst: cropped forward flow estimation map
       bflowEst: cropped backward flow estimation map
       Y_flow: cropped true forward flow map
       Y_flowb: cropped true backward flow map
       Y_occ: cropped true forward occlusion map
       Y_occb: cropped true backward occlusion map
    """

    diffV = np.shape(img1)[1]-sizeV
    diffH = np.shape(img1)[2]-sizeH
                            
    if is_train ==1:
        # random cropping of inputs to size (sizeV, sizeH)
        randV = random.randint(0, diffV)
        randH = random.randint(0, diffH)
    else:
        randV = diffV//2
        randH = diffH//2

    img1 = img1[:,randV:(sizeV+randV),randH:(sizeH+randH),:]
    img2 = img2[:,randV:(sizeV+randV),randH:(sizeH+randH),:]
    sp1 = sp1[:,randV:(sizeV+randV),randH:(sizeH+randH),:]
    sp2 = sp2[:,randV:(sizeV+randV),randH:(sizeH+randH),:]
    Y_mb = Y_mb[:,randV:(sizeV+randV),randH:(sizeH+randH),:]
    Y_mbb = Y_mbb[:,randV:(sizeV+randV),randH:(sizeH+randH),:]
    flowEst = flowEst[:,randV:(sizeV+randV),randH:(sizeH+randH),:]
    bflowEst = bflowEst[:,randV:(sizeV+randV),randH:(sizeH+randH),:]
        
    # revise the occlusion label with the cropping
    Y_occ = crop_occ(Y_occ, Y_flow, randV, randH, sizeV, sizeH)
    Y_occb = crop_occ(Y_occb, Y_flowb, randV, randH, sizeV, sizeH)
    
    return img1, img2, sp1, sp2, Y_mb, Y_mbb, flowEst, bflowEst, Y_occ, Y_occb
            

def crop_occ(gtocc, gtflow, initV, initH, sizeV,  sizeH):
    """
    When cropping, revise occlusion map to consider occlusions that go out of 
        frame in the second frame.
    
    Args:
       gtocc: True occlusion map of size (batch_size, height, width, 1)
       gtflow: True flow map of size (batch_size, height, width, 2)
       (initV, initH): index of top-left pixel of the crop
       (sizeV, sizeH): size of cropped output map

    Returns:
        gtocc_crop: Cropped occlusion label map that appropriately includes 
            pixels that go out of frame in the second cropped image.
    """

    gtocc_crop = gtocc
    
    # Construct base indices which are displaced with the flow
    batch_size, height, width, _ = gtocc.shape
    pos_x = np.tile(range(width), [height * batch_size])
    grid_y = np.tile(np.expand_dims(range(height), 1), [1, width])
    pos_y = np.tile(np.reshape(grid_y, [-1]), [batch_size])
    pos_x = np.reshape(pos_x, [batch_size, height, width]).astype(float)
    pos_y = np.reshape(pos_y, [batch_size, height, width]).astype(float)
    
    # warp the base indices map to the second frame using gtflow
    x0 = pos_x + gtflow[:,:,:,0]
    y0 = pos_y + gtflow[:,:,:,1]
    
    # mark pixels that go out of frame in the second image crop as occlusion
    gtocc_crop[x0>(sizeH+initH-1)] = 1
    gtocc_crop[x0<initH] = 1
    gtocc_crop[y0>(sizeV+initV-1)] = 1
    gtocc_crop[y0<initV] = 1
    
    # crop
    gtocc_crop = gtocc_crop[:,initV:(sizeV+initV),initH:(sizeH+initH),:]
    
    return gtocc_crop
    

class FT3D_Dataset(Sequence):
    """
    Data generator of FlyingThings3D dataset for MONet
    """
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.dataset_root = args.dataset_root
        self.fids = load_ft3d_fids(args.dataset_root, args.is_train)
        self.flowEst_root = args.flowEst_root
        self.is_train = args.is_train
        self.sizeH = args.sizeH
        self.sizeV = args.sizeV
        self.counter = 0
        self.epoch_counter = 0
        self.numY = 6
        self.spN = 2000

        # Total number of batches in each epoch
        self.batch_count = len(self.fids) // self.batch_size    
        # Assigining ID to each data for shuffling at each epoch
        self.indices = np.arange(len(self.fids))

    def __len__(self):
        """number of batches per epoch"""
        return self.batch_count

    def __getitem__(self, idx):
        """ Retrieve sample at idx """
        i = np.sort(self.indices[idx * self.batch_size:(idx+1) * self.batch_size])

        img1, img2, sp1, sp2, Y_mb, Y_mbb, flowEst, bflowEst, Y_flow, Y_flowb,\
            Y_occ, Y_occb = load_ft3ds(self.dataset_root, self.flowEst_root,
                                       self.is_train, self.spN, self.fids, i)
        img1, img2, sp1, sp2, Y_mb, Y_mbb, flowEst, bflowEst, Y_occ, \
            Y_occb = crop_ft3ds(self.sizeV, self.sizeH, img1, img2, sp1, sp2, 
                                Y_mb, Y_mbb, flowEst, bflowEst, Y_flow, 
                                Y_flowb, Y_occ, Y_occb, self.is_train)
    
        # gaussian pyramid inputs
        img1s, sp1s, flowEsts = pyramid_inputs(img1, flowEst, self.spN)
        img2s, sp2s, bflowEsts = pyramid_inputs(img2, bflowEst, self.spN)
                        
        X = [img1]+img1s+[img2]+img2s+[sp1]+sp1s+[sp2]+sp2s+[flowEst]+ \
            flowEsts+[bflowEst]+bflowEsts
        Y = [Y_mb]*self.numY+[Y_occ]*self.numY+[Y_mbb]*self.numY+ \
            [Y_occb]*self.numY+[Y_mb]*(self.numY-1)+[Y_mbb]*(self.numY-1)
        
        self.counter += 1
        return X, Y
                
    def on_epoch_end(self):
        """
        Method called at the end of every epoch.
        """
        self.epoch_counter += 1
        # suffle data along the first dimension
        np.random.shuffle(self.indices)
        return

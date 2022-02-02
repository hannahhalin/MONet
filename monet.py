import numpy as np
import tensorflow as tf
import math
from tensorflow import keras
from keras import backend as K
from keras import initializers
from keras.models import Sequential, Model
from keras.layers import Input, Dropout, BatchNormalization, Activation, Lambda
from keras.layers import concatenate, Concatenate, Multiply, Conv2D, LeakyReLU
from keras.layers import Conv2DTranspose, Add, AveragePooling2D, ReLU
from core_warp import dense_image_warp, images_forward_warp
from loss import photometric_loss_map


def monet(sizeV, sizeH):
    """
        MONet for motion boundary and occlusion detection.
        
        Args:
            sizeV, sizeH - dimension of input image to the network
            
        Returns:
            net - Model for MONet
    """
    
    # Inputs
    images1 = [Input(shape=(sizeV, sizeH, 3), name="img1")]
    images2 = [Input(shape=(sizeV, sizeH, 3), name="img2")]
    sps = [Input(shape=(sizeV, sizeH, 1), name="sp")]
    sps2 = [Input(shape=(sizeV, sizeH, 1), name="sp2")]
    flowEsts = [Input(shape=(sizeV, sizeH, 2), name="flowEst")]
    bflowEsts = [Input(shape=(sizeV, sizeH, 2), name="bflowEst")]

    factor = 1
    for level in range(4):
        factor *= 2
        images1.append(Input(shape=(math.ceil(sizeV/factor), 
                                    math.ceil(sizeH/factor), 3), 
                             name="img1_" + str(level)))
        images2.append(Input(shape=(math.ceil(sizeV/factor), 
                                    math.ceil(sizeH/factor), 3), 
                             name="img2_" + str(level)))
        sps.append(Input(shape=(math.ceil(sizeV/factor), 
                                math.ceil(sizeH/factor), 1), 
                         name="sp_" + str(level)))
        sps2.append(Input(shape=(math.ceil(sizeV/factor), 
                                 math.ceil(sizeH/factor), 1), 
                          name="sp2_" + str(level)))
        flowEsts.append(Input(shape=(math.ceil(sizeV/factor), 
                                     math.ceil(sizeH/factor), 2), 
                              name="flowEst_" + str(level)))
        bflowEsts.append(Input(shape=(math.ceil(sizeV/factor), 
                                      math.ceil(sizeH/factor), 2), 
                               name="bflowEst_" + str(level)))
                        
    costs, bcosts = [], []
    
    # encoder for feature embedding
    fe_model = monet_encoder(sizeV, sizeH)
    fe1 = fe_model(images1)
    fe2 = fe_model(images2)

    # cost volume computation comparing features of img1 with featuers of img2
    factor = 1
    for level in range(4):
        factor *= 2
        cbFunc = cost_block_frame(offset=flowEsts[level+1])
        bcbFunc = cost_block_frame(offset=bflowEsts[level+1])
        costs.append(Lambda(cbFunc, output_shape=(math.ceil(sizeV/factor),
                                                  math.ceil(sizeH/factor), 1),
                            name='corr_layer_'+str(level))([fe1[level],
                                                            fe2[level]]))
        bcosts.append(Lambda(bcbFunc, 
                             output_shape=(math.ceil(sizeV/factor), 
                                           math.ceil(sizeH/factor), 1), 
                             name='bcorr_layer_'+str(level))([fe2[level], 
                                                              fe1[level]]))
        
    # predictor
    mb, occ, mbb, occb, att_mb, att_mbb \
        = monet_predictor(images1, images2, sps, sps2, flowEsts, bflowEsts, 
                          costs, bcosts, height=sizeV, width=sizeH) 

    # model
    net = Model(inputs = images1 + images2 + sps + sps2 + flowEsts + bflowEsts,
                outputs = mb + occ + mbb + occb + att_mb + att_mbb)
    return net


def monet_encoder(height, width, pyramid_levels=4, embedding_dimension=32, 
                  kernel_size=3, encoder_layer_per_block=2, lrelu_alpha=0.1):
    """
        Encoder of MONet to embed mutli-scale features.
        
        Args:
            height - Height of input image
            width - Width of input image
            embedding_dimension - Dimension of embedded features
            pyramid_levels - Number of pyramid levels
            kernel_size - Size of the kernel used in the convolutional layers
            encoder_layer_per_block - Number of layers in each encoder block
            lrelu_alpha - alpha parameter used in leaky relu activation layer.
        
        Returns:
            net - Encoder of MONet for motion boundary detection task.
    """

    def encoding_block(image, channels, block_id, dimension_step):
        y = image
        for layer in range(encoder_layer_per_block):
            layer_id = layer + 1
            y = Conv2D(channels, (kernel_size, kernel_size), padding='same', 
                       name='eblock{}_conv{}'.format(block_id, layer_id))(y)
            y = BatchNormalization(momentum=0.9, epsilon=1e-5, center=False, 
                                   scale=False, 
                                   name='eblock{}_bn{}'.format(block_id, 
                                                               layer_id))(y)
            y = LeakyReLU(alpha=lrelu_alpha, 
                          name='eblock{}_lrelu{}'.format(block_id, 
                                                         layer_id))(y)
        
        y = down_sample_conv(y, channels+dimension_step, kernel_size, 
                             lrelu_alpha, 1, 'eblock{}'.format(block_id))
        f = Dropout(rate=0.3, name='eblock{}_drop_side'.format(block_id))(y)
        f = Conv2D(embedding_dimension, (8,8), padding='same', 
                   name='eblock{}_conv_side'.format(block_id))(f)
        f = BatchNormalization(momentum=0.9, epsilon=1e-5, center=False, 
                               scale=False, 
                               name='eblock{}_bn_side'.format(block_id))(f)
        return y, f
    
    
    # Inputs
    images = [Input(shape=(height, width, 3), name="img1")]
    for level in range(pyramid_levels):
        height, width = math.ceil(height/2), math.ceil(width/2)
        images.append(Input(shape=(height, width, 3), name="img_"+str(level)))
        
    embeddings = []
    dimension = 32
    dimension_step = 16 
    for level in range(pyramid_levels):
        x = images[0] if level == 0 else Lambda(concatenate)([x, 
                                                              images[level]])
        x, feature = encoding_block(x, int(dimension), level+1, dimension_step)
        embeddings.append(feature)
        dimension += dimension_step

    net = Model(inputs=images, outputs=embeddings)
    return net


def monet_predictor(images1, images2, sps, sps2, flowEsts, bflowEsts, costs, 
                    bcosts, height=256, width=448, pyramid_levels=5, 
                    fine_to_coarseI=True, kernel_size=3, start_channels=128, 
                    predictor_layer_per_block=5, lrelu_alpha=0.1, 
                    dropout_rate=0.1):
    """
        Predictor of MONet to estimate multi-scale motion boundaries and 
            occlusions.
        
        Args:
            (images1, images2): Multi-scale input image pairs
            (sps, sps2): Multi-scale superpixel boundaries map
            (flowEsts, bflowEsts): Multi-scale estimated optical flow maps
            (costs, bcosts): Mutli-scale cost blocks
            pyramid_levels: Number of pyramid levels
            (height, width): Dimension of input image to the network
            fine_to_coarseI: False - coarse-to-fine, True - fine-to-coarse
            kernel_size: kernel size of the convolutional layers
            start_channels: initial number of feature channels
            predictor_layer_per_block: number of predictors in each block
            lrelu_alpha: alpha parameter of relu layer
            dropout_rate: dropout rate in dropout layer
            
        Returns:
            (acts_mb, acts_mbb) - Mutli-scale and fusion layer prediction of 
                                      motion boundaries
            (acts_occ, acts_occb) - Mutli-scale and fusion layer prediction of 
                                        occlusions
            (atts_mb, atts_mbb) - Mutli-scale attention maps
    """
    
    def predictor_block(y, y2, block_id, channels, dim_rate, task): 

        def predicting_layer(channels, layer_number):
            layer_id = layer_number + 1
            c = Conv2D(channels, (kernel_size, kernel_size), padding='same', 
                       name='{}_pblock{}_conv{}'.format(task, block_id, 
                                                        layer_id))
            b = None
            d = None
            if layer_number == 0:
                b = BatchNormalization(name='{}_pblock{}_bn{}'.format(
                                       task, block_id, layer_id))
                d = Dropout(rate=dropout_rate, 
                            name='{}_pblock{}_drop{}'.format(
                            task, block_id, layer_id))
            r = LeakyReLU(alpha=lrelu_alpha, name='{}_pblock{}_lrelu{}'.format(
                          task, block_id, layer_id))
            return c,b,d,r
        

        for layer in range(predictor_layer_per_block):
            c,b,d,r = predicting_layer(channels, layer)
            y = c(y)
            y2 = c(y2)
            if b is not None:
                y=b(y)
                y2=b(y2)
            y=r(y)
            y2=r(y2)
            if d is not None:
                y=d(y)
                y2=d(y2)
            
            if layer % 2 == 1:
                channels //= 2
        
        c, ct = side_branch(dim_rate, '{}_pblock{}'.format(task, block_id))
        est = ct(c(y))
        estb = ct(c(y2))
        act = Activation('sigmoid', name='{}_o{}'.format(task, blk_id))(est)
        actb = Activation('sigmoid', name='{}b_o{}'.format(task, blk_id))(estb)

        if fine_to_coarseI:
            if block_id > 1:
                c = Conv2D(channels, (kernel_size, kernel_size), strides=2, 
                           padding='same', 
                           name='{}_pblock{}_conv_down'.format(task, block_id))
                r = LeakyReLU(alpha=lrelu_alpha, 
                              name='{}_pblock{}_lrelu_down'.format(task,
                                                                   block_id))
                y = r(c(y))
                y2 = r(c(y2))
        else:
            if block_id < pyramid_levels:
                ct = Conv2DTranspose(channels, (kernel_size, kernel_size), 
                                     strides=2, padding='same', use_bias=False,
                                     activation=None, 
                                     name='{}_pblock{}_convt_up'.format(
                                     task, block_id))
                r = LeakyReLU(alpha=lrelu_alpha, 
                              name='{}_pblock{}_lrelu_up'.format(task, 
                                                                 block_id))
                y = r(ct(y))
                y2 = r(ct(y2))
        return y, y2, est, act, estb, actb
        
        
    def compute_att_map(input, inputb, task, level, dim_rate=0):
        """Compute attention map.
        Args:
            input, inputb: list of concatenated input features.
            name: Op scope name
        """
        
        def att_layers(task, level):
            conv1 = Conv2D(32, (3, 3), padding='same', 
                           name='{}_att{}_conv0'.format(task, level))
            relu1 = LeakyReLU(alpha=lrelu_alpha, 
                              name='{}_att{}_relu0'.format(task, level))
            conv2 = Conv2D(16, (3, 3), padding='same', 
                           name='{}_att{}_conv1'.format(task, level))
            relu2 = LeakyReLU(alpha=lrelu_alpha, 
                              name='{}_att{}_relu1'.format(task, level))
            conv3 = Conv2D(1, (3, 3), padding='same', 
                           name='{}_att{}_conv2'.format(task, level))
            act = Activation('sigmoid', 
                             name='{}_att{}_sigmoid'.format(task, level))
            act2 = Activation('sigmoid', 
                              name='{}b_att{}_sigmoid'.format(task, level))
            return conv1, relu1, conv2, relu2, conv3, act, act2
            
        conv1, relu1, conv2, relu2, conv3, act, act2  = att_layers(task, level)
        feat = conv3(relu2(conv2(relu1(conv1(input)))))
        featb = conv3(relu2(conv2(relu1(conv1(inputb)))))
                
        c, ct = side_branch(dim_rate, '{}_att{}'.format(task, level), 
                            '_up1', '_up2')
        att_full = act(ct(c(feat)))
        attb_full = act2(ct(c(featb)))
            
        ap = AveragePooling2D(pool_size=(2*dim_rate,2*dim_rate), 
                              strides=dim_rate, padding="same")
        att = ap(att_full)
        attb = ap(attb_full)
            
        return att, attb, att_full, attb_full
                

    def fusion_module(acts, ests, pyramid_levels, name):
        segfuse = Lambda(concatenate)(ests)
        ests.append(Conv2D(1, (1, 1), padding='same', use_bias=False, 
                           activation=None, 
                           kernel_initializer=initializers.Constant(
                                value=1/pyramid_levels),
                           name=name+'_segfuse_conv')(segfuse))
        acts.append(Activation('sigmoid', name=name+'_ofuse')(ests[-1]))
        return acts, ests
        

    ests_mb, acts_mb, ests_mbb, acts_mbb = [], [], [], []
    ests_occ, acts_occ, ests_occb, acts_occb = [], [], [], []
    atts_mb, atts_mbb = [], []
    
    if fine_to_coarseI:
        levels = range(pyramid_levels-1, -1, -1)
        dim_rate = 1
    else:
        levels = range(pyramid_levels)
        dim_rate = 2**(pyramid_levels-1)
                                                    
    flowEsts2_full = Lambda(dense_image_warp, output_shape=(height, width, 2), 
                            name='dense_flow_warp')([bflowEsts[0], 
                                                     flowEsts[0]])
    bflowEst2_full = Lambda(dense_image_warp, output_shape=(height, width, 2), 
                            name='bdense_flow_warp')([flowEsts[0], 
                                                      bflowEsts[0]])
    flowEsts2_full = Lambda(multiply_factor(tf.ones((height, width, 2))*-1), 
                            output_shape=(height, width, 2),
                            name='negate_flow')(flowEsts2_full)
    bflowEst2_full = Lambda(multiply_factor(tf.ones((height, width, 2))*-1), 
                            output_shape=(height, width, 2),
                            name='negate_bflow')(bflowEst2_full)
                                    
    image_warp_full = Lambda(dense_image_warp, 
                             output_shape=(height, width, 3),
                             name='dense_image_warp')([images2[0], 
                                                       flowEsts[0]])
    image_warpb_full = Lambda(dense_image_warp, 
                              output_shape=(height, width, 3),
                              name='bdense_image_warp')([images1[0], 
                                                         bflowEsts[0]])
    warpingError_full = Lambda(photometric_loss_map, 
                               output_shape=(height, width, 3),
                               name='photometric_loss_map')([images1[0], 
                                                             image_warp_full])
    warpingErrorb_full = Lambda(photometric_loss_map, 
                                output_shape=(height, width, 3),
                                name='bphotometric_loss_map')(
                                    [images2[0], image_warpb_full])
                    
    for level in levels:
        img_id = pyramid_levels - level - 1
        blk_id = level + 1
            
        xs_mb = [images1[img_id], images2[img_id]]
        xs_mbb = [images2[img_id], images1[img_id]]
        xs_occ = [images1[img_id], images2[img_id]]
        xs_occb = [images2[img_id], images1[img_id]]        
        xs_mb_att, xs_mbb_att = [], [] # attention map for mb 
        
        thisFlow = flowEsts[img_id]
        thisBFlow = bflowEsts[img_id]
        thisFlow_full = flowEsts[0]
        thisBFlow_full = bflowEsts[0]
        if level != levels[0]:

            # warp for backward motion bundaries map
            x_mb_warp_full, x_mbb_warp_full = Lambda(images_forward_warp, 
                name='image_forward_warp_mb_act_'+str(level))(
                [act_mb, act_mbb, thisFlow_full, thisBFlow_full])
            # down-sampling with strided slice
            mb, mbb, mb_warp, mbb_warp, occ, occb = Lambda(strided_slices_wrap(
                int(dim_rate)), name='strided_slice_'+str(level))([act_mb,
                    act_mbb, x_mb_warp_full, x_mbb_warp_full, act_occ, act_occb
                    ])

            # compute image grad on occlusion map
            act_occb_grad = Lambda(image_gradient, 
                                   output_shape=(height, width, 2),
                                   name='image_gradient_warp_occb_act_'+
                                        str(level))(act_occb)
            act_occ_grad = Lambda(image_gradient, 
                                  output_shape=(height, width, 2),
                                  name='image_gradient_warp_occ_act_'+
                                       str(level))(act_occ)
                                                
            # warp for backward occlusions
            act_occ_grad_warp, act_occb_grad_warp \
                = Lambda(images_forward_warp, 
                         name='image_forward_warp_occ_act_grad_'+
                         str(level))([act_occ_grad, act_occb_grad, 
                                      thisFlow_full, thisBFlow_full])

            # down-sampling with strided slice
            occ_grad, occb_grad, occ_grad_warp, occb_grad_warp \
                = Lambda(strided_slices_wrap(int(dim_rate)), 
                         name='strided_slice_occ_grad'+str(level)
                         )([act_occ_grad, act_occb_grad, act_occ_grad_warp, 
                            act_occb_grad_warp])
                     
            # warp for backward motion bundaries and occlusion feature map
            x_mb_warp, x_mbb_warp, x_occ_warp, x_occb_warp \
                = Lambda(images_forward_warp, 
                         name='image_forward_warp_'+str(level)
                         )([x_mb, x_mbb, x_occ, x_occb, thisFlow, thisBFlow,
                            thisFlow, thisBFlow])    
            
            xs_mb.extend([mb, x_mb])
            xs_mbb.extend([mbb, x_mbb])
            xs_occ.extend([occ, mb, mbb_warp, x_occ, x_mb, x_mbb_warp])
            xs_occb.extend([occb, mbb, mb_warp, x_occb, x_mbb, x_mb_warp])
            xs_mb_att.extend([occ_grad, occb_grad_warp, x_occ, x_occb_warp])
            xs_mbb_att.extend([occb_grad, occ_grad_warp, x_occb, x_occ_warp])

        # down-sampling with strided slice  
        warpingError, warpingErrorb \
            = Lambda(strided_slices_wrap(int(dim_rate)), 
                     name='strided_slice_photometric_loss_'+str(level)
                     )([warpingError_full, warpingErrorb_full])
        flowEsts2, bflowEst2 = Lambda( \
            strided_slices_wrap(int(dim_rate), 
                                tf.ones((math.ceil(height/dim_rate), 
                                math.ceil(width/dim_rate), 2))/dim_rate), 
                                name='strided_slice_flow_warp_' + str(level)
                                )([flowEsts2_full, bflowEst2_full])
           
        xs_mb.extend([thisFlow, flowEsts2, warpingError, sps[img_id], 
                      sps2[img_id]])
        xs_mbb.extend([thisBFlow, bflowEst2, warpingErrorb, sps2[img_id], 
                       sps[img_id]])
        xs_occ.extend([thisFlow, flowEsts2, warpingError, sps[img_id], 
                       sps2[img_id]])
        xs_occb.extend([thisBFlow, bflowEst2, warpingErrorb, sps2[img_id], 
                        sps[img_id]])
        xs_mb_att.extend([sps[img_id], sps2[img_id]])
        xs_mbb_att.extend([sps2[img_id], sps[img_id]])
            
        # cost blocks
        if dim_rate !=1:
            xs_mb_att.extend([costs[img_id-1], bcosts[img_id-1]])
            xs_mbb_att.extend([bcosts[img_id-1], costs[img_id-1]])
            xs_mb.extend([costs[img_id-1], bcosts[img_id-1]])
            xs_mbb.extend([bcosts[img_id-1], costs[img_id-1]])
            xs_occ.extend([costs[img_id-1], bcosts[img_id-1]])
            xs_occb.extend([bcosts[img_id-1], costs[img_id-1]])
            
        # attention module for motion boundaries prediction
        x_mb = Concatenate(axis=3)(xs_mb)
        x_mbb = Concatenate(axis=3)(xs_mbb)
        x_mb_att = Concatenate(axis=3)(xs_mb_att)
        x_mbb_att = Concatenate(axis=3)(xs_mbb_att)
        mb_att, mbb_att, mb_att_full, mbb_att_full \
            = compute_att_map(x_mb_att, x_mbb_att, 'mb', level, int(dim_rate))
        atts_mb.append(mb_att_full)
        atts_mbb.append(mbb_att_full)
        x_mb = Lambda(multiply, name='attention_map_filter_mb'+str(level)
                      )([x_mb, mb_att]) # x_mb = x_mb * mb_att
        x_mbb = Lambda(multiply, name='attention_map_filter_mbb'+str(level)
                       )([x_mbb, mbb_att]) # x_mbb = x_mbb * mbb_att

        # motion boundaries predictor block
        x_mb, x_mbb, est_mb, act_mb, est_mbb, act_mbb = predictor_block(
            x_mb, x_mbb, blk_id, start_channels, int(dim_rate), 'mb')
        ests_mb.insert(0, est_mb)
        acts_mb.insert(0, act_mb)
        ests_mbb.insert(0, est_mbb)
        acts_mbb.insert(0, act_mbb)
        
        # occlusions predictor block
        x_occ = Concatenate(axis=3)(xs_occ)
        x_occb = Concatenate(axis=3)(xs_occb)
        x_occ, x_occb, est_occ, act_occ, est_occb, act_occb = predictor_block(
            x_occ, x_occb, blk_id, start_channels, int(dim_rate), 'occ')
        ests_occ.insert(0, est_occ)
        acts_occ.insert(0, act_occ)
        ests_occb.insert(0, est_occb)
        acts_occb.insert(0, act_occb)
            
        if fine_to_coarseI:
            dim_rate *= 2
        else:
            dim_rate /= 2
    
    # fusion module
    acts_mb, ests_mb = fusion_module(acts_mb, ests_mb, pyramid_levels, 'mb')
    acts_mbb, ests_mbb = fusion_module(acts_mbb, ests_mbb, pyramid_levels, 
                                       'mbb')
    acts_occ, ests_occ = fusion_module(acts_occ, ests_occ, pyramid_levels, 
                                       'occ')
    acts_occb, ests_occb = fusion_module(acts_occb, ests_occb, pyramid_levels, 
                                         'occb')
    
    return acts_mb, acts_occ, acts_mbb, acts_occb, atts_mb, atts_mbb
    
    
def cost_block_frame(offset=None, search_range=5, dillation=1, minCostV=1):
    """
        Returns the function for cost volume computation comparing features 
            of img1 and img2.
        
        Args:
            search_range - search range of features of img2 to compare 
                                for each feature of img1
            offset - initial estimation of flow used to find the local 
                        neighborhood in the second image
        
        Returns:
            cost_block - Function for cost block computation comparing 
                            features of img1 and img2.
    """
    
    def cost_block(fs):
        """
            Build cost block for associating a feature from img1 with its 
                corresponding features in img2. Adapted from tfoptflow by 
                philferriere (https://github.com/philferriere/tfoptflow).
            
            Args:
                fs: embedded features of two consecutive image frame inputs
                
            Returns:
                cost_blk: cost block for associating a feature from img1 with 
                              its corresponding features in img2.
        """
        
        c1, c2 = fs
        padded_c2 = tf.pad(c2, [[0, 0], [search_range, search_range], 
                           [search_range, search_range], [0, 0]])
        _, h, w, _ = tf.unstack(tf.shape(c1))

        cost_blk = []
        for y in range(-search_range, search_range+1):
            for x in range(-search_range, search_range+1):
                extra_offset = np.array([[[[x*dillation, y*dillation]]]])
                new_offset = offset + extra_offset
                slice = dense_image_warp([c2, new_offset])
                
                cost = tf.sqrt(tf.reduce_sum(tf.square(c1-slice), axis=3, 
                                             keepdims=True))
                cost_blk.append(cost)
                
        cost_blk = tf.concat(cost_blk, axis=3)
        cost_blk = tf.nn.leaky_relu(cost_blk, alpha=0.1)
        if minCostV ==1:
            cost_blk = tf.reduce_min(cost_blk, axis=3, keepdims=True)
        return cost_blk
    return cost_block


def strided_slice_wrap(stride):
    def strided_slice(x):
        xs = tf.strided_slice(x, [0,0,0,0], tf.shape(x), strides=[1,stride,stride,1])
        return xs 
    return strided_slice


def strided_slices_wrap(stride, factor=1):
    def strided_slices(xs):
        x = tf.concat(xs, axis=0) 
        x = tf.strided_slice(x, [0,0,0,0], tf.shape(x), 
                             strides=[1,stride,stride,1])
        x = tf.multiply(x, factor)
        xs2 = tf.split(x, num_or_size_splits=len(xs), axis=0)
        for i in range(len(xs)):
            xs2[i].set_shape((xs[0].shape[0], xs[0].shape[1]//stride, 
                              xs[0].shape[2]//stride, xs[0].shape[3]))
        return xs2
    return strided_slices


def image_gradient(image):
    dx, dy = tf.image.image_gradients(image)
    grad = tf.concat([dx, dy], axis=-1)
    return grad


def split_batch_wrap(num_input):
    def split_batch(xs):
        return tf.split(xs, num_or_size_splits=num_input, axis=0)
    return split_batch


def multiply_factor(factor):
    def multiply(x):
        return tf.multiply(x,factor)
    return multiply
    
    
def multiply(xs):
    x1, x2 = xs
    return tf.multiply(x1,x2)

                                                                
def gradientAdd(gradients) :
    gradient1, gradient2 = gradients
    return tf.expand_dims(gradient1[:,:,:,0]+gradient1[:,:,:,1]+
                          gradient2[:,:,:,0]+gradient2[:,:,:,1],axis=3)
             
            
def down_sample_conv(x, channels, kernel_size, lrelu_alpha, use_act, nameL):
    x = Conv2D(channels, (kernel_size, kernel_size), strides=2, padding='same',
               name=nameL+'_conv_down')(x)
    x = BatchNormalization(center=False, scale=False, name=nameL+'_bn_down')(x)
    if use_act==1:
        x = LeakyReLU(alpha=lrelu_alpha, name=nameL+'_lrelu_down')(x)
    return x


def side_branch(factor, nameI, nameIa='_convMB5', nameIb='_convMB6'):
    """
        Side output layers used to output multi-scale estimations from the 
            intermediate layers of the predictor
        
        Inputs:
            x - features obtained from the intermediate layers of the predictor
            factor - factor used for up-sampling multi-scale estimations to the 
                        original resolution of the MB-Net input
            nameI - prefix to add to the layer name
        
        Returns:
            x - multi-scale estimation from the intermediate layers of the 
                    predictor
    """
    c = Conv2D(1, (1, 1), activation=None, padding='same', name=nameI+nameIa)
    kernel_size = (2*factor, 2*factor)
    ct = Conv2DTranspose(1, kernel_size, strides=factor, padding='same', 
                         use_bias=False, activation=None, name=nameI+nameIb)
    return c, ct

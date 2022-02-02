import tensorflow as tf
from keras import backend as K

def getMyJointLosses(pyramid_levels=5, weight=1):
    """
    Joint losses to train MONet.
        
    Args:
        pyramid_levels: Number of pyramid levels or multi-scale predictions
        weight: weight of the losses
    Returns:
        losses: losses for training 
        weights: weight to be applied to each of the losses
    
    """

    focalLoss = focal_loss_sigmoid()
        
    losses = {}
    for thisY in range(pyramid_levels):
        losses['mb_o'+str(thisY+1)] = focalLoss
        losses['mbb_o'+str(thisY+1)] = focalLoss
        losses['occ_o'+str(thisY+1)] = focalLoss
        losses['occb_o'+str(thisY+1)] = focalLoss
        losses['mb_att'+str(thisY)+'_sigmoid'] = focalLoss
        losses['mbb_att'+str(thisY)+'_sigmoid'] = focalLoss
    losses['mb_ofuse'] = focalLoss
    losses['mbb_ofuse'] = focalLoss
    losses['occ_ofuse'] = focalLoss
    losses['occb_ofuse'] = focalLoss

    weights = [weight]*(pyramid_levels+1)*2*2 + [weight/10]*pyramid_levels*2

    return losses, weights
    

def focal_loss_sigmoid(alpha=0.25, gamma=2.):

    def focal_loss(y_true, y_pred):
        """
        Focal Loss
        """
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1-K.epsilon())
        L = tf.reduce_sum(-y_true*(1-alpha)*((1-y_pred)**gamma)* \
                          tf.math.log(y_pred), [1,2,3])- \
                          tf.reduce_sum((1-y_true)*alpha*(y_pred**gamma)* \
                          tf.math.log(1-y_pred), [1,2,3])
        cost = tf.reduce_mean(L)
        return cost

    return focal_loss


def photometric_loss_map(inputs, mask=None):
    """
    Photometric Loss
    """
    with tf.compat.v1.variable_scope('photometric_loss_map'):
        return charbonnier_loss_map(inputs[0]-inputs[1], mask, beta=255)


def charbonnier_loss_map(x, mask=None, truncate=None, alpha=0.45,
                         beta=1.0, epsilon=0.001):
    """Compute the generalized charbonnier loss of the difference tensor x.
            All positions times mask are not taken into account.
            Adapted from UnFlow written by simonmeister
            (https://github.com/simonmeister/UnFlow).

    Args:
        x: a tensor of shape [num_batch, height, width, channels].
        mask: a mask of shape [num_batch, height, width, mask_channels],
            where mask channels must be either 1 or the same number as
            the number of channels of x.
    Returns:
        loss as tf.float32
    """
    with tf.compat.v1.variable_scope('charbonnier_loss_map'):
        batch, height, width, channels = tf.unstack(tf.shape(x))

        error = tf.pow(tf.square(x * beta) + tf.square(epsilon), alpha)

        if mask is not None:
            error = tf.multiply(mask, error)

        if truncate is not None:
            error = tf.minimum(error, truncate)

        return error
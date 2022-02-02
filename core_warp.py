"""
core_warp.py

Direct and reverse warping of feature maps using their flow map.

Reference - [tfoptflow](https://github.com/philferriere/tfoptflow) by philferriere.
"""

from __future__ import absolute_import, division, print_function

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
import tensorflow as tf


def _interpolate_bilinear(grid,
                          query_points,
                          name='interpolate_bilinear',
                          indexing='ij'):
    """Similar to Matlab's interp2 function.

    Finds values for query points on a grid using bilinear interpolation.

    Args:
      grid: a 4-D float `Tensor` of shape `[batch, height, width, channels]`.
      query_points: a 3-D float `Tensor` of N points with shape `[batch, N, 2]`.
      name: a name for the operation (optional).
      indexing: whether the query points are specified as row and column (ij),
        or Cartesian coordinates (xy).

    Returns:
      values: a 3-D `Tensor` with shape `[batch, N, channels]`

    Raises:
      ValueError: if the indexing mode is invalid, or if the shape of the inputs
        invalid.
    """
    if indexing != 'ij' and indexing != 'xy':
        raise ValueError('Indexing mode must be \'ij\' or \'xy\'')

    with ops.name_scope(name):
        grid = ops.convert_to_tensor(grid)
        query_points = ops.convert_to_tensor(query_points)
        shape = array_ops.unstack(array_ops.shape(grid))
        if len(shape) != 4:
            msg = 'Grid must be 4 dimensional. Received: '
            raise ValueError(msg + str(shape))

        batch_size, height, width, channels = shape
        query_type = query_points.dtype
        query_shape = array_ops.unstack(array_ops.shape(query_points))
        grid_type = grid.dtype

        if len(query_shape) != 3:
            msg = ('Query points must be 3 dimensional. Received: ')
            raise ValueError(msg + str(query_shape))

        _, num_queries, _ = query_shape

        alphas = []
        floors = []
        ceils = []

        index_order = [0, 1] if indexing == 'ij' else [1, 0]
        unstacked_query_points = array_ops.unstack(query_points, axis=2)

        for dim in index_order:
            with ops.name_scope('dim-' + str(dim)):
                queries = unstacked_query_points[dim]

                size_in_indexing_dimension = shape[dim + 1]

                # max_floor is size_in_indexing_dimension - 2 so that max_floor + 1
                # is still a valid index into the grid.
                max_floor = math_ops.cast(size_in_indexing_dimension - 2, query_type)
                min_floor = constant_op.constant(0.0, dtype=query_type)
                floor = math_ops.minimum(
                    math_ops.maximum(min_floor, math_ops.floor(queries)), max_floor)
                int_floor = math_ops.cast(floor, dtypes.int32)
                floors.append(int_floor)
                ceil = int_floor + 1
                ceils.append(ceil)

                # alpha has the same type as the grid, as we will directly use alpha
                # when taking linear combinations of pixel values from the image.
                alpha = math_ops.cast(queries - floor, grid_type)
                min_alpha = constant_op.constant(0.0, dtype=grid_type)
                max_alpha = constant_op.constant(1.0, dtype=grid_type)
                alpha = math_ops.minimum(math_ops.maximum(min_alpha, alpha), max_alpha)

                # Expand alpha to [b, n, 1] so we can use broadcasting
                # (since the alpha values don't depend on the channel).
                alpha = array_ops.expand_dims(alpha, 2)
                alphas.append(alpha)

        flattened_grid = array_ops.reshape(grid,
                                           [batch_size * height * width, channels])
        batch_offsets = array_ops.reshape(
            math_ops.range(batch_size) * height * width, [batch_size, 1])

        # This wraps array_ops.gather. We reshape the image data such that the
        # batch, y, and x coordinates are pulled into the first dimension.
        # Then we gather. Finally, we reshape the output back. It's possible this
        # code would be made simpler by using array_ops.gather_nd.
        def gather(y_coords, x_coords, name):
            with ops.name_scope('gather-' + name):
                linear_coordinates = batch_offsets + y_coords * width + x_coords
                gathered_values = array_ops.gather(flattened_grid, linear_coordinates)
                return array_ops.reshape(gathered_values,
                                         [batch_size, num_queries, channels])

        # grab the pixel values in the 4 corners around each query point
        top_left = gather(floors[0], floors[1], 'top_left')
        top_right = gather(floors[0], ceils[1], 'top_right')
        bottom_left = gather(ceils[0], floors[1], 'bottom_left')
        bottom_right = gather(ceils[0], ceils[1], 'bottom_right')

        # now, do the actual interpolation
        with ops.name_scope('interpolate'):
            interp_top = alphas[1] * (top_right - top_left) + top_left
            interp_bottom = alphas[1] * (bottom_right - bottom_left) + bottom_left
            interp = alphas[0] * (interp_bottom - interp_top) + interp_top

        return interp


def dense_image_warp(inputs, name='dense_image_warp'):
    """Image warping using per-pixel flow vectors.

    Apply a non-linear warp to the image, where the warp is specified by a dense
    flow field of offset vectors that define the correspondences of pixel values
    in the output image back to locations in the  source image. Specifically, the
    pixel value at output[b, j, i, c] is
    images[b, j - flow[b, j, i, 0], i - flow[b, j, i, 1], c].

    The locations specified by this formula do not necessarily map to an int
    index. Therefore, the pixel value is obtained by bilinear
    interpolation of the 4 nearest pixels around
    (b, j - flow[b, j, i, 0], i - flow[b, j, i, 1]). For locations outside
    of the image, we use the nearest pixel values at the image boundary.


    Args:
      image: 4-D float `Tensor` with shape `[batch, height, width, channels]`.
      flow: A 4-D float `Tensor` with shape `[batch, height, width, 2]`.
      name: A name for the operation (optional).

      Note that image and flow can be of type tf.half, tf.float32, or tf.float64,
      and do not necessarily have to be the same type.

    Returns:
      A 4-D float `Tensor` with shape`[batch, height, width, channels]`
        and same type as input image.

    Raises:
      ValueError: if height < 2 or width < 2 or the inputs have the wrong number
                  of dimensions.
    """
    with ops.name_scope(name):
        image, flow = inputs
        batch_size, height, width, channels = array_ops.unstack(array_ops.shape(image))
        
        # The flow is defined on the image grid. Turn the flow into a list of query
        # points in the grid space.
        grid_x, grid_y = array_ops.meshgrid(
            math_ops.range(width), math_ops.range(height))
        stacked_grid = math_ops.cast(
            array_ops.stack([grid_y, grid_x], axis=2), flow.dtype)
        batched_grid = array_ops.expand_dims(stacked_grid, axis=0)
        flow_yx = tf.unstack(flow, axis=-1)
        flow_yx = tf.stack([flow_yx[1], flow_yx[0]], axis=-1)
        query_points_on_grid = batched_grid + flow_yx
        query_points_flattened = array_ops.reshape(query_points_on_grid,
                                                   [batch_size, height * width, 2])
        # Compute values at the query points, then reshape the result back to the
        # image grid.
        interpolated = _interpolate_bilinear(image, query_points_flattened)
        # Previous implementation makes the last channel None
        # interpolated = array_ops.reshape(interpolated,
        #                                  [batch_size, height, width, channels])
        interpolated = array_ops.reshape(interpolated,
                                         array_ops.shape(image))
        return interpolated


def images_forward_warp(inputs, name='forward_warp'):
    """Performs a forward warp of an image using the predicted flow.

    Args:
        im: Batch of images. [num_batch, height, width, channels]
        flow: Batch of flow vectors. [num_batch, height, width, 2]
    Returns:
        warped: transformed image of the same shape as the input image.
    Note:
        The holes will be zero.
    """
    with tf.compat.v1.variable_scope(name+'image_forward_warp'):
        ims, flows = inputs[:len(inputs)//2], inputs[len(inputs)//2:]
        im = tf.concat(ims, axis=0) 
        flow = tf.concat(flows, axis=0) 
        num_batch, height, width, channels = tf.unstack(tf.shape(im))

        max_x = tf.cast(width - 1, 'int32')
        max_y = tf.cast(height - 1, 'int32')
        zero = tf.zeros([], dtype='int32')

        # We have to flatten our tensors to vectorize the interpolation
        im_flat = tf.reshape(im, [-1, channels])
        flow_flat = tf.reshape(flow, [-1, 2])

        # Floor the flow, as the final indices are integers
        # The fractional part is used to control the bilinear interpolation.
        flow_floor = tf.to_int32(tf.floor(flow_flat))
        bilinear_weights = flow_flat - tf.floor(flow_flat)

        # Construct base indices which are displaced with the flow
        pos_x = tf.tile(tf.range(width), [height * num_batch])
        grid_y = tf.tile(tf.expand_dims(tf.range(height), 1), [1, width])
        pos_y = tf.tile(tf.reshape(grid_y, [-1]), [num_batch])

        x = flow_floor[:, 0]
        y = flow_floor[:, 1]
        xw = bilinear_weights[:, 0]
        yw = bilinear_weights[:, 1]

        # Compute splat weights for 4 adjacent pixels. The propagated pixel is
        # splatted into the 4 pixels according to the weights.
        # expand to num_batch * height * width x 1 for broadcasting in add_n below
        wa = tf.expand_dims((1 - xw) * (1 - yw), 1) # top left pixel
        wb = tf.expand_dims((1 - xw) * yw, 1) # bottom left pixel
        wc = tf.expand_dims(xw * (1 - yw), 1) # top right pixel
        wd = tf.expand_dims(xw * yw, 1) # bottom right pixel

        x0 = pos_x + x
        x1 = x0 + 1
        y0 = pos_y + y
        y1 = y0 + 1

        x0 = tf.clip_by_value(x0, zero, max_x)
        x1 = tf.clip_by_value(x1, zero, max_x)
        y0 = tf.clip_by_value(y0, zero, max_y)
        y1 = tf.clip_by_value(y1, zero, max_y)

        dim1 = width * height
        batch_offsets = tf.range(num_batch) * dim1
        base_grid = tf.tile(tf.expand_dims(batch_offsets, 1), [1, dim1])
        base = tf.reshape(base_grid, [-1])

        # Find the location of pixels that get some assigned values.
        base_y0 = base + y0 * width
        base_y1 = base + y1 * width
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        # Weighted sum with normalized weights
        indices = tf.expand_dims(tf.concat([idx_a, idx_b, idx_c, idx_d], axis=0), 1)

        # Calculate the sum of weight
        weights_sum = tf.scatter_nd(
            indices,
            tf.squeeze(tf.concat([wa, wb, wc, wd], axis=0), axis=1),
            tf.shape(im_flat)[:1], name="sum_weights")
        useful_weights = tf.gather(params=weights_sum, indices=indices)

        # The useful weights are at least 1e-6 for numerical stability
        inv_weights = 1 / tf.clip_by_value(useful_weights, clip_value_min=1e-6, clip_value_max=1e10)

        warped_flat = tf.scatter_nd(
            indices,
            (tf.concat([wa*im_flat, wb*im_flat, wc*im_flat, wd*im_flat], axis=0) *
            inv_weights),
            tf.shape(im_flat), name="warp_image_with_normalized_weights")

        warped = tf.reshape(warped_flat, [num_batch, height, width, channels])

        warped_split = tf.split(warped, num_or_size_splits=len(inputs)//2, axis=0)
        for i in range(len(inputs)//2):
            warped_split[i].set_shape(inputs[0].shape)
        return warped_split
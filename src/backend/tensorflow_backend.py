"""Extensions to the TensorFlow back-end for Keras."""
from keras import backend as K
from keras.backend.tensorflow_backend import tf
from keras.backend.tensorflow_backend import _preprocess_conv2d_input
from keras.backend.tensorflow_backend import _preprocess_padding


def confusion_matrix(y_true, y_pred, num_classes=None):
    """
    Compute a confusion matrix from predictions and ground truths.

    Args:
        y_true: the ground truth labels
        y_pred: the predicted labels
        num_classes: the optional number of classes. if not provided, the
                     labels are assumed to be in [0, max]
    Returns:
        a confusion matrix computed based on y_true and y_pred

    """
    return tf.confusion_matrix(y_true, y_pred, num_classes=num_classes)


def pool2d_argmax(x: 'Tensor', pool_size: tuple,
    strides: tuple=(1, 1),
    padding: str='valid',
    data_format: str=None
) -> tuple:
    """
    2D Pooling that returns indexes too.

    Args:
        x: Tensor or variable.
        pool_size: tuple of 2 integers.
        strides: tuple of 2 integers.
        padding: string, `"same"` or `"valid"`.
        data_format: string, `"channels_last"` or `"channels_first"`.

    Returns:
        A tensor, result of 2D pooling.

    Raises:
        ValueError: if `data_format` is neither `"channels_last"` or `"channels_first"`.

    """
    # get the normalized data format
    data_format = K.common.normalize_data_format(data_format)
    # pre-process the input tensor
    x, tf_data_format = _preprocess_conv2d_input(x, data_format)
    padding = _preprocess_padding(padding)
    # update strides and pool size based on data format
    if tf_data_format == 'NHWC':
        strides = (1,) + strides + (1,)
        pool_size = (1,) + pool_size + (1,)
    else:
        strides = (1, 1) + strides
        pool_size = (1, 1) + pool_size
    # get the values and the indexes from the max pool operation
    x, idx = tf.nn.max_pool_with_argmax(x, pool_size, strides, padding=padding)
    # update shapes if necessary
    if data_format == 'channels_first' and tf_data_format == 'NHWC':
        # NHWC -> NCHW
        x = tf.transpose(x, (0, 3, 1, 2))
        # NHWC -> NCHW
        idx = tf.transpose(idx, (0, 3, 1, 2))

    return x, idx


def unpool2d_argmax(x: 'Tensor', idx: 'Tensor', pool_size: tuple) -> 'Tuple':
    """
    Un-pooling layer to complement pool2d_argmax.

    Args:
        x: input Tensor or variable
        idx: index matching the shape of x from the pooling operation
        pool_size: the pool_size used by the pooling operation

    Returns:
        an un-pooled version of x using indexes in idx

    Notes:
        follow the issue here for updates on this functionality:
        https://github.com/tensorflow/tensorflow/issues/2169

    """
    # get the input shape of the tensor
    in_s = K.shape(x)
    # get the output shape of the tensor
    out_s = [in_s[0], in_s[1] * pool_size[0], in_s[2] * pool_size[1], in_s[3]]

    # get the size of the batch-wise flattened output matrix
    flat_output_shape = [out_s[0], out_s[1] * out_s[2] * out_s[3]]

    # create an index over the batches
    batch_range = K.arange(K.cast(in_s[0], 'int64'))
    batch_range = K.reshape(batch_range, shape=[in_s[0], 1, 1, 1])
    # create a ones tensor in the shape of index
    batch_idx = K.ones_like(idx) * batch_range
    batch_idx = K.reshape(batch_idx, (-1, 1))
    # create a complete index
    index = K.reshape(idx, (-1, 1))
    index = K.concatenate([batch_idx, index])

    # flatten the inputs and un-pool
    pool = K.flatten(x)
    ret = tf.scatter_nd(index, pool, K.cast(flat_output_shape, 'int64'))
    # reshape the output in the correct shape
    ret = K.reshape(ret, out_s)

    # update the integer shape of the Keras Tensor
    in_s = K.int_shape(x)
    ret_s = [in_s[0], in_s[1] * pool_size[0], in_s[2] * pool_size[1], in_s[3]]
    ret.set_shape(ret_s)

    return ret


# explicitly define the outward facing API of this module
__all__ = [
    confusion_matrix.__name__,
    pool2d_argmax.__name__,
    unpool2d_argmax.__name__,
]

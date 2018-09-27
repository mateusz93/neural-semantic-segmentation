"""Extensions to the TensorFlow back-end for Keras."""
from keras import backend as K
from keras.backend.tensorflow_backend import tf
from keras.backend.tensorflow_backend import _preprocess_conv2d_input
from keras.backend.tensorflow_backend import _preprocess_padding


def confusion_matrix(y_true, y_pred, num_classes=None, weights=None):
    """
    Compute a confusion matrix from predictions and ground truths.

    Args:
        y_true: the ground truth labels
        y_pred: the predicted labels
        num_classes: the optional number of classes. if not provided, the
                     labels are assumed to be in [0, max]
        weights: Optional Tensor whose rank is either 0, or the same rank as
                 labels, and must be broadcast-able to labels (i.e., all
                 dimensions must be either 1, or the same as the corresponding
                 labels dimension). Use weights of 0 to mask values.

    Returns:
        a confusion matrix computed based on y_true and y_pred

    """
    return tf.confusion_matrix(y_true, y_pred,
        num_classes=num_classes,
        weights=weights
    )


def pool2d_argmax(x, pool_size,
    strides=(1, 1),
    padding='valid',
    data_format=None,
    pool_mode='max'
) -> tuple:
    """
    2D Pooling that returns indexes too.

    Args:
        x: Tensor or variable.
        pool_size: tuple of 2 integers.
        strides: tuple of 2 integers.
        padding: string, `"same"` or `"valid"`.
        data_format: string, `"channels_last"` or `"channels_first"`.
        pool_mode: string, `"max"` or `"avg"`.

    Returns:
        A tensor, result of 2D pooling.

    Raises:
        ValueError: if `data_format` is neither `"channels_last"` or `"channels_first"`.
        ValueError: if `pool_mode` is neither `"max"` or `"avg"`.

    """
    data_format = K.common.normalize_data_format(data_format)

    x, tf_data_format = _preprocess_conv2d_input(x, data_format)
    padding = _preprocess_padding(padding)
    if tf_data_format == 'NHWC':
        strides = (1,) + strides + (1,)
        pool_size = (1,) + pool_size + (1,)
    else:
        strides = (1, 1) + strides
        pool_size = (1, 1) + pool_size

    if pool_mode == 'max':
        x, idx = tf.nn.max_pool_with_argmax(x, pool_size, strides, padding=padding)
    elif pool_mode == 'avg':
        raise NotImplementedError('no support for avg pooling with index')
    else:
        raise ValueError('Invalid pool_mode: ' + str(pool_mode))

    if data_format == 'channels_first' and tf_data_format == 'NHWC':
        x = tf.transpose(x, (0, 3, 1, 2))  # NHWC -> NCHW
        idx = tf.transpose(idx, (0, 3, 1, 2))  # NHWC -> NCHW

    return x, idx


def unpool2d_argmax(x, idx, pool_size):
    """
    Un-pooling layer after pool2d_argmax.

    Args:
        x: Tensor or variable
        idx: index matching the shape of x
        pool_size: the pool_size used by the pooling operation

    Returns:
        an un-pooled version of x using indexes in idx

    Notes:
        follow the issue here for updates on this functionality:
        https://github.com/tensorflow/tensorflow/issues/2169

    """
    in_s = K.shape(x)
    out_s = [in_s[0], in_s[1] * pool_size[0], in_s[2] * pool_size[1], in_s[3]]

    flat_input_size = K.prod(in_s)
    flat_output_shape = [out_s[0], out_s[1] * out_s[2] * out_s[3]]

    b_shape = [in_s[0], 1, 1, 1]
    b_range = K.reshape(K.arange(K.cast(out_s[0], 'int64')), shape=b_shape)
    b = K.ones_like(idx) * b_range
    b1 = K.reshape(b, [flat_input_size, 1])
    ind_ = K.reshape(idx, [flat_input_size, 1])
    ind_ = K.concatenate([b1, ind_], 1)

    pool = K.reshape(x, [flat_input_size])
    ret = tf.scatter_nd(ind_, pool, shape=K.cast(flat_output_shape, 'int64'))
    ret = K.reshape(ret, out_s)

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

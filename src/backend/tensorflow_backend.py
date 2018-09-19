"""Extensions to the TensorFlow backend for Keras."""
from keras import backend as K
from keras.backend.tensorflow_backend import tf
from keras.backend.tensorflow_backend import _preprocess_conv2d_input
from keras.backend.tensorflow_backend import _preprocess_padding
from keras.backend.tensorflow_backend import _to_tensor


def categorical_crossentropy(target, output, weights=None, axis=-1):
    """Categorical crossentropy between an output tensor and a target tensor.
    # Arguments
        target: A tensor of the same shape as `output`.
        output: A tensor resulting from a softmax
            (unless `from_logits` is True, in which
            case `output` is expected to be the logits).
        weights:
        axis: Int specifying the channels axis. `axis=-1`
            corresponds to data format `channels_last`,
            and `axis=1` corresponds to data format
            `channels_first`.
    # Returns
        Output tensor.
    # Raises
        ValueError: if `axis` is neither -1 nor one of
            the axes of `output`.
    """
    if weights is None:
        return K.categorical_crossentropy(target, output,
            from_logits=False,
            axis=axis
        )
    output_dimensions = list(range(len(output.get_shape())))
    if axis != -1 and axis not in output_dimensions:
        raise ValueError(
            '{}{}{}'.format(
                'Unexpected channels axis {}. '.format(axis),
                'Expected to be -1 or one of the axes of `output`, ',
                'which has {} dimensions.'.format(len(output.get_shape()))))

    # scale preds so that the class probas of each sample sum to 1
    output /= tf.reduce_sum(output, axis, True)
    # manual computation of crossentropy
    _epsilon = _to_tensor(K.epsilon(), output.dtype.base_dtype)
    output = tf.clip_by_value(output, _epsilon, 1. - _epsilon)
    return - tf.reduce_sum(target * tf.log(output) * weights, axis)


def pool2d_argmax(x, pool_size,
    strides=(1, 1),
    padding='valid',
    data_format=None,
    pool_mode='max'
) -> tuple:
    """2D Pooling that returns indexes too.

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
    pool2d_argmax.__name__,
    unpool2d_argmax.__name__,
]

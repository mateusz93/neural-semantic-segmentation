"""Extensions to the TensorFlow backend for Keras."""
from keras.backend.tensorflow_backend import tf
from keras.backend.tensorflow_backend import _preprocess_conv2d_input
from keras.backend.tensorflow_backend import _preprocess_padding
from keras.backend.common import normalize_data_format


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
    data_format = normalize_data_format(data_format)

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
        # TODO: implement or find implementation for average pooling with index
        raise NotImplementedError('TensorFlow is missing the required method')
    else:
        raise ValueError('Invalid pool_mode: ' + str(pool_mode))

    if data_format == 'channels_first' and tf_data_format == 'NHWC':
        x = tf.transpose(x, (0, 3, 1, 2))  # NHWC -> NCHW
        idx = tf.transpose(idx, (0, 3, 1, 2))  # NHWC -> NCHW

    return x, idx


# explicitly define the outward facing API of this module
__all__ = ['pool2d_argmax']

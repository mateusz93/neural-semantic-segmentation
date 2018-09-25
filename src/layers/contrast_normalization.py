"""A Keras layer to normalize image inputs using different techniques."""
import keras.backend as K
from keras.layers.core import Layer


def conv2d(inputs, kernel):
    """
    Convolve over the inputs using the given kernel.

    Args:
        inputs: the inputs to convolve
        kernel: the kernel to use in the convolution

    Returns:
        the output from the convolution operation over inputs using the kernel

    """
    # convolve over the inputs using the kernel with same shape padding
    channels = []
    for i in range(K.int_shape(inputs)[-1]):
        channels += [K.conv2d(inputs[..., i:i+1], kernel, padding='same')]
    return K.mean(K.concatenate(channels, axis=-1), axis=-1, keepdims=True)


def normal_kernel(kernel_size, mean=1.0, scale=0.05):
    """
    Return a new Gaussian RGB kernel with given layer size.

    Args:
        kernel_size: the size of the kernel
        mean: the mean for the Gaussian randomness
        scale: the scale for the Gaussian randomness

    Returns:
        a Gaussian RGB kernel normalized to sum to 1

    """
    # create the kernel shape with square kernel, 1 expected input channel,
    # and 1 filter in total (i.e., 1 output channel)
    kernel_shape = (kernel_size, kernel_size, 1, 1)
    # create a random normal variable with given mean and scale
    kernel = K.random_normal(kernel_shape, mean=mean, stddev=scale)
    # normalize the values to ensure the sum of the filter is 1
    kernel = kernel / K.sum(kernel)
    return kernel


class ContrastNormalization(Layer):
    """A Keras layer to normalize image inputs using different techniques."""

    def __init__(self,
        kernel_size=9,
        method='l2',
        depth_radius=5,
        bias=1.0,
        alpha=0.0001,
        beta=0.75,
        **kwargs
    ):
        """
        Initialize a new contrast normalization layer.

        Args:
            kernel_size: the size of the kernel to use in Gaussian kernels
            method: the technique to use as one of
            - 'l2': Local L2 Normalization
            - 'lcn': Local Contrast Normalization (LeCunn)
            depth_radius:
            bias:
            alpha:
            beta:

        Returns:
            None

        """
        # ensure the provided method is valid
        methods = {
            'l2': self.local_l2_normalization,
            'lcn': self.local_contrast_normalization,
            'lrn': self.local_response_normalization,
            'slrn': self.spatial_local_response_normalization
        }
        if method not in methods.keys():
            msg = 'method must be in: {}'.format(repr(set(methods.keys())))
            raise ValueError(msg)
        # ensure the kernel size is legal
        if not isinstance(kernel_size, (int, float)):
            raise TypeError('kernel_size must be a numeric value')
        elif kernel_size < 1:
            raise ValueError('kernel_size must be >= 1')
        # call the super constructor
        super(ContrastNormalization, self).__init__(**kwargs)
        # store instance variables
        self.kernel_size = int(kernel_size)
        # setup the call(self, inputs) method based on the provided
        self.call = methods[method]
        self.depth_radius = depth_radius
        self.bias = bias
        self.alpha = alpha
        self.beta = beta

    def local_l2_normalization(self, inputs):
        """Local L2 Normalization."""
        kernel = normal_kernel(self.kernel_size)
        sigma = K.sqrt(conv2d(K.square(inputs), kernel))
        sigma += K.cast(K.equal(sigma, 0), K.floatx())
        return inputs / sigma

    def local_response_normalization(self, inputs):
        """Local Response Normalization."""
        from keras.backend.tensorflow_backend import tf
        return tf.nn.lrn(inputs,
            depth_radius=self.depth_radius ,
            bias=self.bias,
            alpha=self.alpha,
            beta=self.beta,
        )

    def spatial_local_response_normalization(self, inputs):
        """Spatial Local Response Normalization"""
        squared = K.square(inputs)
        kernel = normal_kernel(self.depth_radius)
        squared_sum = conv2d(squared, kernel)
        return inputs / ((self.bias + self.alpha * squared_sum) ** self.beta)

    def local_contrast_normalization(self, inputs):
        """LeCunn Local Contrast Normalization."""
        kernel = normal_kernel(self.kernel_size)
        v = inputs - conv2d(inputs, kernel)
        sigma = K.sqrt(conv2d(K.square(v), kernel))
        mean = K.mean(sigma, axis=[1, 2])
        mean = K.expand_dims(K.expand_dims(mean, axis=1), axis=1)
        return v / K.maximum(mean, sigma)


# explicitly define the outward facing API of this module
__all__ = [ContrastNormalization.__name__]

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
    return K.conv2d(inputs, kernel, padding='same')


def normal_kernel(kernel_size, mean=10, scale=1, input_channels=3):
    """
    Return a new Gaussian RGB kernel with given layer size.

    Args:
        kernel_size: the size of the kernel
        mean: the mean for the Gaussian randomness
        scale: the scale for the Gaussian randomness
        input_channels: the number of input channels into the kernel

    Returns:
        a Gaussian RGB kernel normalized to sum to 1

    """
    # create the kernel shape with square kernel, 1 expected input channel,
    # and 1 filter in total (i.e., 1 output channel)
    kernel_shape = (kernel_size, kernel_size, 1, 1)
    # create a random normal variable with given mean and scale
    kernel = K.random_normal_variable(kernel_shape, mean=mean, scale=scale)
    # normalize the values to ensure the sum of the filter is 1
    kernel = kernel / K.sum(kernel)
    # repeat along the input channel axis if input channels is more than 1
    if input_channels > 1:
        kernel = K.repeat_elements(kernel, input_channels, axis=-2)
    return kernel


class ContrastNormalization(Layer):
    """A Keras layer to normalize image inputs using different techniques."""

    def __init__(self, kernel_size=9, method='l2', **kwargs):
        """
        Initialize a new contrast normalization layer.

        Args:
            kernel_size: the size of the kernel to use in Gaussian kernels
            method: the technique to use as one of
            - 'l2': Local L2 Normalization
            - 'lcn': Local Contrast Normalization (LeCunn)

        Returns:
            None

        """
        # ensure the kernel size is legal
        if not isinstance(kernel_size, (int, float)):
            raise TypeError('kernel_size must be a numeric value')
        elif kernel_size < 1:
            raise ValueError('kernel_size must be >= 1')
        # ensure the provided method is valid
        legal_methods = {'l2', 'lcn'}
        if method not in legal_methods:
            msg = 'method must be in: {}'.format(repr(legal_methods))
            raise ValueError(msg)
        # call the super constructor
        super(ContrastNormalization, self).__init__(**kwargs)
        # store instance variables
        self.kernel_size = int(kernel_size)
        self.method = method
        # setup the call(self, inputs) method based on the provided
        if self.method == 'l2':
            self.call = self.local_l2_normalization
        elif self.method == 'lcn':
            self.call = self.local_contrast_normalization

    def local_l2_normalization(self, inputs):
        """Local L2 Normalization."""
        kernel = normal_kernel(self.kernel_size)
        sigma = K.sqrt(conv2d(K.square(inputs), kernel))
        sigma = K.maximum(sigma, 1.0)
        return inputs / sigma

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

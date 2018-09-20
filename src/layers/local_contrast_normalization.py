import keras.backend as K
from keras.layers.core import Layer


class LocalContrastNormalization(Layer):

    def __init__(self, kernel_size=9, **kwargs):
        super(LocalContrastNormalization, self).__init__(**kwargs)
        self.kernel_size = kernel_size

    def call(self, inputs):
        filter_shape = (self.kernel_size, self.kernel_size, 1, 1)
        filters = K.random_normal_variable(filter_shape, mean=10, scale=1)
        filters = filters / K.sum(filters)
        filters = K.repeat_elements(filters, 3, axis=-2)

        v = inputs - K.conv2d(inputs, filters, padding='same')

        sigma = K.sqrt(K.conv2d(K.square(v), filters, padding='same'))
        mean = K.mean(sigma, axis=[1, 2])
        mean = K.expand_dims(K.expand_dims(mean, axis=1), axis=1)

        return v / K.maximum(mean, sigma)

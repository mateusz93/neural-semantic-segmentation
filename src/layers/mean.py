"""A layer to calculate the mean over an axis."""
from keras import backend as K
from keras.layers import Layer


class Mean(Layer):
    """A layer to calculate the mean over an axis."""

    def __init__(self, axis=-1, **kwargs):
        """
        Initialize a new mean layer.

        Args:
            axis: the axis to calculate the mean over

        Returns:
            None

        """
        # initialize with the super constructor
        super(Mean, self).__init__(**kwargs)
        # store the instance variables of this layer
        self.axis = axis

    def call(self, inputs, **kwargs):
        """
        Forward pass through the layer.

        Args:
            inputs: the tensor to calculate the mean over axis of
            **kwargs: extra keyword arguments

        Returns:
            the input tensor stacked self.n times along axis 1

        """
        return K.mean(inputs, axis=self.axis)


# explicitly define the outward facing API of this module
__all__ = [Mean.__name__]

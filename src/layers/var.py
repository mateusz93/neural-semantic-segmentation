"""A layer to calculate the variance over an axis."""
from keras import backend as K
from keras.layers import Layer


class Var(Layer):
    """A layer to calculate the variance over an axis."""

    def __init__(self, axis=-1, **kwargs):
        """
        Initialize a new variance layer.

        Args:
            axis: the axis to calculate the variance over

        Returns:
            None

        """
        # initialize with the super constructor
        super(Var, self).__init__(**kwargs)
        # store the instance variables of this layer
        self.axis = axis

    def get_output_shape_for(self, input_shape):
        """
        Return the output shape of the layer for given input shape.

        Args:
            input_shape: the input shape to transform to output shape

        Returns:
            the output shape as a function of the input shape (1 extra dim)

        """
        shape_size = len(input_shape)
        if self.axis < 0:
            axis = shape_size + self.axis
        else:
            axis = self.axis
        return input_shape[:axis] + input_shape[axis+1:]

    def call(self, inputs, **kwargs):
        """
        Forward pass through the layer.

        Args:
            inputs: the tensor to calculate the variance over axis of
            **kwargs: extra keyword arguments

        Returns:
            the input tensor stacked self.n times along axis 1

        """
        return K.var(inputs, axis=self.axis)


# explicitly define the outward facing API of this module
__all__ = [Var.__name__]

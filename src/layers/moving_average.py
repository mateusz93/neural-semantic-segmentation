"""A layer to calculate an Exponential Moving Average (EMA)."""
from keras import backend as K
from keras.layers import Layer


class MovingAverage(Layer):
    """A layer to calculate an Exponential Moving Average (EMA)."""

    def __init__(self, momentum=0.9, **kwargs):
        """
        Initialize a new repeat tensor layer.

        Args:
            momentum: the momentum of the moving average
            kwargs: keyword arguments for the super constructor

        Returns:
            None

        """
        # initialize with the super constructor
        super(MovingAverage, self).__init__(**kwargs)
        # store the instance variables of this layer
        self.momentum = momentum

    def build(self, input_shape):
        """Build the layer with given input shape."""
        # create a variable to keep the moving average in
        self.average = self.add_weight(
            shape=(1, ) + input_shape[1:],
            name='average',
            initializer='zeros',
            trainable=False
        )
        # mark the layer as built
        self.built = True

    def call(self, inputs):
        """
        Forward pass through the layer.

        Args:
            inputs: the tensor to perform the stack operation on
            training: whether the layer is in the training phase

        Returns:
            the input tensor stacked self.n times along axis 1

        """
        # update the moving average
        self.average = self.momentum * inputs + (1 - self.momentum) * self.average
        return self.average


# explicitly define the outward facing API of this module
__all__ = [MovingAverage.__name__]

"""A layer to calculate a moving average with momentum."""
from keras import backend as K
from keras.layers import Layer


class MovingAverage(Layer):
    """A layer to calculate a moving average with momentum."""

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

    def call(self, inputs, training=None):
        """
        Forward pass through the layer.

        Args:
            inputs: the tensor to perform the stack operation on
            training: whether the layer is in the training phase

        Returns:
            the input tensor stacked self.n times along axis 1

        """
        # no moving average if training
        if training in {0, False}:
            return inputs
        # create a variable to keep the moving average in
        self.average = K.zeros(K.int_shape(inputs)[1:])
        # create an update operation for the moving average from inputs
        update = K.moving_average_update(self.average, inputs[0], self.momentum)
        # add the moving average update (conditional on the inputs)
        self.add_update(update, inputs)
        # return the inputs if training, moving average if testing
        return K.in_train_phase(inputs, K.expand_dims(self.average, axis=0), training=training)


# explicitly define the outward facing API of this module
__all__ = [MovingAverage.__name__]

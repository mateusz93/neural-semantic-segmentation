"""A layer to perform Monte Carlo simulation on a function."""
from keras import backend as K
from keras.layers import Layer


class MonteCarlo(Layer):
    """A layer to perform Monte Carlo simulation on a function."""

    def __init__(self, function, simulations, **kwargs):
        """
        Initialize a new repeat tensor layer.

        Args:
            function: the layer or model to simulate using Monte Carlo
            simulations: the number of times to call the function on inputs
            kwargs: keyword arguments for the super constructor

        Returns:
            None

        """
        # initialize with the super constructor
        super(MonteCarlo, self).__init__(**kwargs)
        # store the instance variables of this layer
        self.function = function
        self.simulations = simulations

    def compute_output_shape(self, *args, **kwargs):
        """Return the output shape of this layer."""
        return self.function.output_shape + (self.simulations, )

    def call(self, inputs, **kwargs):
        """
        Forward pass through the layer.

        Args:
            inputs: the tensor to perform the stack operation on
            **kwargs: extra keyword arguments

        Returns:
            the input tensor stacked self.n times along axis 1

        """
        # collect outputs for each simulation
        outputs = [self.function(inputs) for _ in range(self.simulations)]
        # return the average over all the outputs
        return K.stack(outputs, axis=-1)


# explicitly define the outward facing API of this module
__all__ = [MonteCarlo.__name__]

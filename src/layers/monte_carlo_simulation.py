"""A layer to perform Monte Carlo simulation on a function."""
from keras.layers import Layer
from keras.layers import Average


class MonteCarloSimulation(Layer):
    """A layer to perform Monte Carlo simulation on a function."""

    def __init__(self, layer, simulations, **kwargs):
        """
        Initialize a new repeat tensor layer.

        Args:
            layer: the layer or model to simulate using Monte Carlo
            simulations: the number of times to call the function on inputs
            kwargs: keyword arguments for the super constructor

        Returns:
            None

        """
        # initialize with the super constructor
        super(MonteCarloSimulation, self).__init__(**kwargs)
        # store the instance variables of this layer
        self.layer = layer
        self.simulations = simulations

    def compute_output_shape(self, *args, **kwargs):
        """Return the output shape of this layer."""
        return self.layer.output_shape

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
        outputs = [self.layer(inputs) for _ in range(self.simulations)]
        # return the average over all the outputs
        return Average()(outputs)

    @property
    def trainable_weights(self):
        """Return the trainable weights of the layer."""
        return self.layer.trainable_weights

    @property
    def non_trainable_weights(self):
        """Return the non-trainable weights of the layer."""
        return self.layer.non_trainable_weights

    def get_weights(self):
        """Return the weights of the layer."""
        return self.layer.get_weights()

    def set_weights(self, weights):
        """Set the weights of the layer."""
        return self.layer.set_weights(weights)

    def load_weights(self, weights_file):
        """Load the weights for this layer from a file."""
        return self.layer.load_weights(weights_file)


# explicitly define the outward facing API of this module
__all__ = [MonteCarloSimulation.__name__]

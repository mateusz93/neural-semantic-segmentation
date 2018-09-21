"""A layer to perform dropout in both training and testing time."""
from keras.layers import Dropout


class MonteCarloDropout(Dropout):
    """A layer to perform dropout in both training and testing time."""

    def call(self, inputs, **kwargs):
        """
        Forward pass through the layer.

        Args:
            inputs: the tensor to perform the dropout operation on
            **kwargs: extra keyword arguments

        Returns:
            the input with a random number of values dropped

        """
        return super(MonteCarloDropout, self).call(inputs, training=True)


# explicitly define the outward facing API of this module
__all__ = [MonteCarloDropout.__name__]

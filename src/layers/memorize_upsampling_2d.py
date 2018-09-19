"""A 2D up-sampling layer that uses indexes from memorized pooling."""
from keras.layers import UpSampling2D
from ..backend.tensorflow_backend import unpool2d_argmax


class MemorizedUpsampling2D(UpSampling2D):
    """A 2D up-sampling layer that uses indexes from memorized pooling."""

    def __init__(self, *args, idx, **kwargs):
        """
        Initialize a new up-sampling layer using memorized down-sample index.

        Args:
            idx: the memorized index form pool2d_argmax

        Returns:
            None

        """
        super(MemorizedUpsampling2D, self).__init__(*args, **kwargs)
        self.idx = idx

    def call(self, inputs):
        return unpool2d_argmax(inputs, self.idx, self.size)


# explicitly define the outward facing API of this module
__all__ = [MemorizedUpsampling2D.__name__]

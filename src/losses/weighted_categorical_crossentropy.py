"""A Keras implementation of weighted categorical cross entropy loss."""
from ..backend.tensorflow_backend import categorical_crossentropy


def build_weighted_categorical_crossentropy(weights):
    def _categorical_crossentropy(y_true, y_pred):
        return categorical_crossentropy(y_true, y_pred, weights)
    return _categorical_crossentropy


# explicitly define the outward facing API of this module
__all__ = [build_weighted_categorical_crossentropy.__name__]

"""Loss functions for the project."""
from .weighted_categorical_crossentropy import build_weighted_categorical_crossentropy


# explicitly define the outward facing API of this package
__all__ = [build_weighted_categorical_crossentropy.__name__]

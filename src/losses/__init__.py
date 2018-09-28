"""Loss functions for the project."""
from .categorical_crossentropy import build_categorical_crossentropy


# explicitly define the outward facing API of this package
__all__ = [build_categorical_crossentropy.__name__]

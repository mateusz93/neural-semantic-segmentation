"""This package provides methods to load the CamVid dataset into memory."""
from .generator import data_generators
from .visualize import plot


# explicitly define the outward facing API of this package
__all__ = [
    data_generators.__name__,
    plot.__name__,
]

"""This package provides methods to load the CamVid dataset into memory."""
from .create_segmented_y import create_segmented_y
from .generator import data_generators
from .visualize import plot


# explicitly define the outward facing API of this package
__all__ = [
    create_segmented_y.__name__,
    data_generators.__name__,
    plot.__name__,
]

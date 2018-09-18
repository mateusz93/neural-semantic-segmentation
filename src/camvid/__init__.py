"""This package provides methods to load the CamVid dataset into memory."""
from .camvid import CamVid
from .create_segmented_y import create_segmented_y
from .visualize import plot


# explicitly define the outward facing API of this package
__all__ = [
    CamVid.__name__,
    create_segmented_y.__name__,
    plot.__name__,
]

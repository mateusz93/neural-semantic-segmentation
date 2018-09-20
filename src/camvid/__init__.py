"""This package provides methods to load the CamVid dataset into memory."""
from .camvid import CamVid
from .plot import plot


# explicitly define the outward facing API of this package
__all__ = [
    CamVid.__name__,
    plot.__name__,
]

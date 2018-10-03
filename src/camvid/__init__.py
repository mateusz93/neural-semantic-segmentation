"""This package provides methods to load the CamVid dataset into memory."""
from .camvid import CamVid
from .load_mapping import load_mapping
from .plot import plot


# explicitly define the outward facing API of this package
__all__ = [
    CamVid.__name__,
    load_mapping.__name__,
    plot.__name__,
]

"""A package for loading label data mappers."""
from .load_label_metadata import load_label_metadata


# explicitly define the outward facing API of this package
__all__ = [
    load_label_metadata.__name__,
]

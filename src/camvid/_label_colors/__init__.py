"""A package for loading label data mappers."""
from .segmentation_to_discrete import SegmentationToDiscreteTransformer
from .segmentation_to_onehot import SegmentationToOnehotTransformer


# explicitly define the outward facing API of this package
__all__ = [
    SegmentationToDiscreteTransformer.__name__,
    SegmentationToOnehotTransformer.__name__,
]

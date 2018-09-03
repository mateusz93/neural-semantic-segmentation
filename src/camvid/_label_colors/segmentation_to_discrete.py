"""A class to map segmentation images to discrete label values."""
import numpy as np
from .load_label_map import load_label_map


class SegmentationToDiscreteTransformer(object):
    """A transformer to convert pixel images to discrete label grids."""

    def __init__(self) -> None:
        """
        Initialize a new segmentation image to discrete mapper."""
        label_map = load_label_map()
        self._mapping = label_map['encoding']
        self.labels = label_map['labels']
        self._demap = np.vectorize({v: k for k, v in self._mapping.items()}.get)

    def __len__(self) -> int:
        """Return the length of the mapping."""
        return len(self._mapping)

    def map(self, img: np.ndarray):
        """
        Convert the segment image to a discrete grid.

        Args:
            img: the image to convert to a discrete grid

        Returns:
            the segmented image as a 2D grid using the internal mapping

        """
        # create an empty discrete image in the size of the input
        discrete = np.empty(shape=img.shape[:-1], dtype=int)
        discrete[:] = np.nan
        # iterate over the segmentation colors in the mapping
        for rgb, label in self._mapping.items():
            discrete[(img == rgb).all(-1)] = label

        return discrete

    def unmap(self, discrete: np.ndarray):
        """
        Un-map an input grid of discrete values.

        Args:
            discrete: the 2D grid of discrete values

        Returns:
            the unmapped image from the discrete values

        """
        # convert the discrete mapping back to original form using _demap
        img = np.array(self._demap(discrete))
        # move the RGB channel to the last value, as is tradition
        img = np.moveaxis(img, 0, -1)

        return img


# explicitly define the outward facing API of this module
__all__ = [SegmentationToDiscreteTransformer.__name__]

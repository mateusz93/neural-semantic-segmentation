"""A class to map segmentation images to one-hot label vectors."""
import numpy as np
from .segmentation_to_discrete import SegmentationToDiscreteTransformer


class SegmentationToOnehotTransformer(SegmentationToDiscreteTransformer):
    """A transformer to convert pixel images to one-hot vector grids."""

    def map(self, img):
        """
        Convert the segment image to a one-hot vector grid.

        Args:
            img: the image to convert to a one-hot vector grid

        Returns:
            the segmented image as a 2D grid using the internal mapping

        """
        # convert the image to a discrete 2D grid
        img = super().map(img)
        # create the 3D grid for the one-hot vectors
        onehot = np.zeros(shape=(*img.shape, len(self._mapping)), dtype=int)
        # fancy index the tensor with one hot values
        for label in self._mapping.values():
            onehot[img == label, label] = True

        return onehot

    def unmap(self, probabilities):
        """
        Un-map an input grid of one-hot vectors.

        Args:
            discrete: the 3D grid of one-hot vectors

        Returns:
            the unmapped image from the one-hot values

        """
        # convert to a discrete 2D grid using arg-max to select the highest
        # probability value
        discrete = probabilities.argmax(axis=-1)
        # return from the super (discrete) un-map function
        return super().unmap(discrete)


# explicitly define the outward facing API of this module
__all__ = [SegmentationToOnehotTransformer.__name__]

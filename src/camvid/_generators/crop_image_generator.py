"""An Image Generator extension to crop images to a given size."""
import numpy as np
from keras.preprocessing.image import ImageDataGenerator


def _random_crop(tensor: 'numpy.ndarray', image_size: tuple):
    """
    Return a random crop of a tensor.

    Args:
        tensor: the tensor to get a random crop from
        image_size: the size of the cropped box to return

    Returns:
        a random crop of the tensor with shape image_size

    """
    # extract the dimensions of the image
    h = tensor.shape[0]
    w = tensor.shape[1]
    # generate the random crop height dimensions
    h0 = np.random.randint(0, h - image_size[0])
    h1 = h0 + image_size[0]
    # generate the random crop width dimensions
    w0 = np.random.randint(0, w - image_size[1])
    w1 = w0 + image_size[1]

    return tensor[h0:h1, w0:w1]


class CropImageDataGenerator(ImageDataGenerator):
    """An Image Generator extension to crop images to a given size."""

    def __init__(self, *args, image_size=None, **kwargs) -> None:
        """
        Create a new Segment Image Data generator.

        Args:
            args: positional arguments for the ImageDataGenerator super class
            image_size: the image size to crop to
            kwargs: keyword arguments for the ImageDataGenerator super class

        Returns:
            None

        """
        if image_size is not None and not isinstance(image_size, tuple):
            raise TypeError('image_size should be of type: tuple')
        super().__init__(*args, **kwargs)
        self.image_size = image_size

    def apply_transform(self, *args, **kwargs):
        """Apply a transform to the input tensor with given parameters."""
        # get the batch from the super transformer first
        batch = super().apply_transform(*args, **kwargs)
        # map this batch of items to output dimension
        if self.image_size is not None:
            return _random_crop(batch, self.image_size)

        return batch

    def flow_from_directory(self, *args, **kwargs):
        """Create a directory iterator to load from."""
        # get the directory iterator from the super call
        iterator = super().flow_from_directory(*args, **kwargs)
        # change the output dimension of the iterator to support the new
        # number of channels defined by the transformers length
        if self.image_size is not None:
            iterator.image_shape = (*self.image_size, iterator.image_shape[-1])
        return iterator


# explicitly define the outward facing API of this module
__all__ = [CropImageDataGenerator.__name__]

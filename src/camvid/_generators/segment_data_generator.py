"""An Image Generator extension to map RGB images to segmentation vectors."""
from .crop_image_generator import CropImageDataGenerator


class SegmentImageDataGenerator(CropImageDataGenerator):
    """An image generator for segment images that need converted."""

    def __init__(self, *args,
        transformer=None,
        **kwargs
    ) -> None:
        """
        Create a new Segment Image Data generator.

        Args:
            args: positional arguments for the ImageDataGenerator super class
            transformer: the transformer to convert RGB to discrete or one-hot
            kwargs: keyword arguments for the ImageDataGenerator super class

        Returns:
            None

        """
        # raise an error if the required keyword argument is missing
        if transformer is None:
            raise TypeError('transformer should be of type: Callable')
        # pass all arguments to the super constructor
        super().__init__(*args, **kwargs)
        # store the transformer in self
        self.transformer = transformer

    def apply_transform(self, *args, **kwargs):
        """Apply a transform to the input tensor with given parameters."""
        # get the batch from the super transformer first
        batch = super().apply_transform(*args, **kwargs)
        # map this batch of items to output dimension
        return self.transformer.map(batch)

    def flow_from_directory(self, *args, **kwargs):
        """Create a directory iterator to load from."""
        # get the directory iterator from the super call
        iterator = super().flow_from_directory(*args, **kwargs)
        # change the output dimension of the iterator to support the new
        # number of channels defined by the transformers length
        iterator.image_shape = (*iterator.image_shape[:-1], len(self.transformer))
        return iterator


# explicitly define the outward facing API of this module
__all__ = [SegmentImageDataGenerator.__name__]

"""A class for interacting with the CamVid data."""
import ast
import os
import numpy as np
import pandas as pd
from ._create_segmented_y import create_segmented_y
from ._generators import CropImageDataGenerator
from ._generators import CropNumpyDataGenerator
from ._generators import repeat_generator


class CamVid(object):
    """An instance of a CamVid dataset."""

    def __init__(self,
        mapping: dict=None,
        x_repeats: int=0,
        y_repeats: int=0,
        target_size: tuple=(720, 960),
        crop_size: tuple=(224, 224),
        horizontal_flip: bool=False,
        vertical_flip: bool=False,
        batch_size: int=3,
        shuffle: bool=True,
        seed: int=1,
    ) -> None:
        """
        Initialize a new CamVid dataset instance.

        Args:
            y: the directory name with the y label data
            x_repeats: the number of times to repeat the output of x generator
            y_repeats: the number of times to repeat the output of y generator
            target_size: the image size of the dataset
            crop_size: the size to crop images to. if None, apply no crop
            horizontal_flip: whether to randomly flip images horizontally
            vertical_flip whether to randomly flip images vertically
            batch_size: the number of images to load per batch
            shuffle: whether to shuffle images in the dataset
            seed: the random seed to use for the generator

        Returns:
            None

        """
        # get the directory this file is in to locate X
        this_dir = os.path.dirname(os.path.abspath(__file__))
        # locate the X and y directories
        self._x = os.path.join(this_dir, 'X')
        self._y = create_segmented_y(mapping)
        # store remaining keyword arguments
        self.x_repeats = x_repeats
        self.y_repeats = y_repeats
        self.target_size = target_size
        self.crop_size = crop_size
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        # create a vectorized method to map discrete codes to RGB pixels
        self._unmap = np.vectorize(self.discrete_to_rgb_map.get)

    @property
    def n(self) -> int:
        """Return the number of training classes in this dataset."""
        return len(self.metadata['code'].unique())

    @property
    def class_weights(self) -> dict:
        """Return a dictionary of class weights keyed by discrete label."""
        weights = pd.read_csv(os.path.join(self._y, 'weights.csv'), index_col=0)
        # calculate the frequency of each class
        freq = weights['pixels'] / weights['pixels_total']
        # calculate the weights as the median frequency divided by all freq
        return (freq.median() / freq).values

    def data_gen_args(self, context: str) -> dict:
        """
        Return the keyword arguments for creating a new data generator.

        Args:
            context: the context for the call (i.e., train for training)

        Returns:
            a dictionary of keyword arguments to pass to DataGenerator.__init__

        """
        # return for training
        if context == 'train':
            return dict(
                horizontal_flip=self.horizontal_flip,
                vertical_flip=self.vertical_flip,
                image_size=self.crop_size
            )
        # return for validation / testing (i.e., inference)
        return dict(image_size=self.crop_size)

    def flow_args(self, context: str) -> dict:
        """
        Return the keyword arguments for flowing from a data generator.

        Args:
            context: the context for the call (i.e., train for training)

        Returns:
            a dictionary of keyword arguments to pass to flow_from_directory

        """
        # return for training
        if context == 'train':
            return dict(
                batch_size=self.batch_size,
                class_mode=None,
                target_size=self.target_size,
                shuffle=self.shuffle,
                seed=self.seed
            )
        # return for validation / testing (i.e., inference)
        return dict(
            batch_size=1,
            class_mode=None,
            target_size=self.target_size,
            seed=self.seed
        )

    @property
    def metadata(self) -> pd.DataFrame:
        """Return the metadata associated with this dataset."""
        return pd.read_csv(os.path.join(self._y, 'metadata.csv'))

    def _discrete_dict(self, col: str) -> dict:
        """
        Return a dictionary mapping discrete codes to values in another column.

        Args:
            col: the name of the column to map discrete code values to

        Returns:
            a dictionary mapping unique codes to values in the given column

        """
        return self.metadata[['code', col]].set_index('code').to_dict()[col]

    @property
    def discrete_to_rgb_map(self) -> dict:
        """Return a dictionary mapping discrete codes to RGB pixels."""
        rgb_draw = self._discrete_dict('rgb_draw')
        # convert the strings in the RGB draw column to tuples
        return {k: ast.literal_eval(v) for (k, v) in rgb_draw.items()}

    @property
    def discrete_to_label_map(self) -> dict:
        """Return a dictionary mapping discrete codes to RGB pixels."""
        return self._discrete_dict('label_used')

    def unmap(self, y_discrete: np.ndarray) -> np.ndarray:
        """
        Un-map a one-hot vector y frame to the target RGB values.

        Args:
            y_discrete: the one-hot vector to convert to an RGB image

        Returns:
            an RGB encoding of the one-hot input tensor

        """
        return np.stack(self._unmap(y_discrete.argmax(axis=-1)), axis=-1)

    def generators(self) -> dict:
        """Return a dictionary with both training and validation generators."""
        # the dictionary to hold generators by key value (training, validation)
        generators = dict()
        # iterate over the generator subsets
        for subset in ['train', 'val', 'test']:
            # create generators to load images (X) and NumPy tensors (y)
            x_g = CropImageDataGenerator(**self.data_gen_args(subset))
            y_g = CropNumpyDataGenerator(**self.data_gen_args(subset))
            # get the path for the subset of data
            _x = os.path.join(self._x, subset)
            _y = os.path.join(self._y, subset)
            # combine X and y generators into a single generator with repeats
            generators[subset] = repeat_generator(
                x_g.flow_from_directory(_x, **self.flow_args(subset)),
                y_g.flow_from_directory(_y, **self.flow_args(subset)),
                x_repeats=self.x_repeats,
                y_repeats=self.y_repeats,
            )

        return generators


# explicitly define the outward facing API of this module
__all__ = [CamVid.__name__]

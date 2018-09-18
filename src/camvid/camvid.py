"""A class for interacting with the CamVid data."""
import os
from ast import literal_eval as make_tuple
import pandas as pd
import numpy as np
from ._generators import CropImageDataGenerator
from ._generators import CropNumpyDataGenerator


class CamVid(object):
    """An instance of a CamVid dataset."""

    def __init__(self,
        y_dir: str,
        target_size: tuple=(720, 960),
        crop_size: tuple=(224, 224),
        horizontal_flip: bool=False,
        vertical_flip: bool=True,
        validation_split: float=0.3,
        batch_size: int=3,
        shuffle: bool=True,
        seed: int=1,
    ) -> None:
        """
        Initialize a new CamVid dataset instance.

        Args:
            y_dir: the directory name with the y label data
            target_size: the image size of the dataset
            crop_size: the size to crop images to. if None, apply no crop
            horizontal_flip: whether to randomly flip images horizontally
            vertical_flip whether to randomly flip images vertically
            validation_split: the size of the validation set in [0, 1]
            batch_size: the number of images to load per batch
            shuffle: whether to shuffle images in the dataset
            seed: the random seed to use for the generator

        Returns:
            None

        """
        # get the directory this file is in to locate X and y
        this_dir = os.path.dirname(os.path.abspath(__file__))
        # locate the X and y directories
        self.x_dir = os.path.join(this_dir, 'X')
        self.y_dir = os.path.join(this_dir, y_dir)
        # store remaining keyword arguments
        self.target_size = target_size
        self.crop_size = crop_size
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.validation_split = validation_split
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        # create a vectorized method to map discrete codes to RGB pixels
        self._unmap = np.vectorize(self.discrete_to_rgb_map.get)

    @property
    def n(self) -> int:
        """Return the number of training classes in this dataset."""
        return int(self.y_dir.split('_')[-1])

    @property
    def data_gen_args(self) -> dict:
        """Return the keyword arguments for creating a new data generator."""
        return dict(
            horizontal_flip=self.horizontal_flip,
            vertical_flip=self.vertical_flip,
            validation_split=self.validation_split,
            image_size=self.crop_size
        )

    @property
    def flow_args(self) -> dict:
        """Return the keyword arguments for flowing from a data generator."""
        return dict(
            class_mode=None,
            target_size=self.target_size,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            seed=self.seed
        )

    @property
    def metadata(self) -> pd.DataFrame:
        """Return the metadata associated with this dataset."""
        return pd.read_csv(os.path.join(self.y_dir, 'metadata.csv'))

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
        rgb_draw = self._discrete_dict('rgb_draw').items()
        # convert the strings in the rgb draw column to tuples
        return {k: make_tuple(v) for (k, v) in rgb_draw}

    @property
    def discrete_to_label_map(self) -> dict:
        """Return a dictionary mapping discrete codes to RGB pixels."""
        return self._discrete_dict('label_used')

    def unmap(self, y: np.ndarray) -> np.ndarray:
        """
        Un-map a one-hot vector y frame to the target RGB values.

        Args:
            y: the one-hot vector to convert to an RGB image

        Returns:
            an RGB encoding of the one-hot input tensor

        """
        return np.stack(self._unmap(y.argmax(axis=-1)), axis=-1)

    def generators(self) -> dict:
        """Return a dictionary with both training and validation generators."""
        # create the RAW image data generator
        x_data = CropImageDataGenerator(**self.data_gen_args)
        # create the segmentation data generator as a One-Hot tensor
        y_data = CropNumpyDataGenerator(**self.data_gen_args)
        # the dictionaries to hold generators by key value
        generators = {}
        # iterate over the subsets in the generators
        for subset in ['training', 'validation']:
            x = x_data.flow_from_directory(self.x_dir, subset=subset, **self.flow_args)
            y = y_data.flow_from_directory(self.y_dir, subset=subset, **self.flow_args)
            # zip the X and y generators into a single generator
            generators[subset] = zip(x, y)

        return generators


# explicitly define the outward facing API of this module
__all__ = [CamVid.__name__]

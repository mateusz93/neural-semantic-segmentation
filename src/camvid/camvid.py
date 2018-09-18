"""A class for interacting with the CamVid data."""
import os
from ast import literal_eval as make_tuple
import pandas as pd
import numpy as np
from ._generators import CropImageDataGenerator
from ._generators import CropNumpyDataGenerator


# the directory housing this file.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# the directory to load the X images from
X_DIR = os.path.join(THIS_DIR, 'X')


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
    ):
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
        self.y_dir = os.path.join(THIS_DIR, y_dir)
        self.target_size = target_size
        self.crop_size = crop_size
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.validation_split = validation_split
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self._unmap = np.vectorize(self.discrete_to_rgb_map.get)

    @property
    def n(self):
        """Return the number of training classes in this dataset."""
        return int(self.y_dir.split('_')[-1])

    @property
    def data_gen_args(self):
        """Return the keyword arguments for creating a new data generator."""
        return dict(
            horizontal_flip=self.horizontal_flip,
            vertical_flip=self.vertical_flip,
            validation_split=self.validation_split,
            image_size=self.crop_size
        )

    @property
    def flow_args(self):
        """Return the keyword arguments for flowing from a data generator."""
        return dict(
            class_mode=None,
            target_size=self.target_size,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            seed=self.seed
        )

    @property
    def metadata(self):
        """Return the metadata associated with this dataset."""
        return pd.read_csv(os.path.join(self.y_dir, 'metadata.csv'))

    @property
    def discrete_to_rgb_map(self):
        """Return a dictionary mapping discrete codes to RGB pixels."""
        df = self.metadata[['code', 'rgb_draw']].set_index('code')
        pixel_map = df.to_dict()['rgb_draw']
        return {k: make_tuple(v) for (k, v) in pixel_map.items()}

    @property
    def discrete_to_label_map(self):
        """Return a dictionary mapping discrete codes to RGB pixels."""
        df = self.metadata[['code', 'label_used']].set_index('code')
        pixel_map = df.to_dict()['label_used']
        return {k: v for (k, v) in pixel_map.items()}

    def unmap(self, y):
        """
        Un-map a one-hot vector y frame to the target RGB values.

        Args:
            y: the one-hot vector to convert to an RGB image

        Returns:
            an RGB encoding of the one-hot input tensor

        """
        return np.stack(self._unmap(y.argmax(axis=-1)), axis=-1)

    def generators(self):
        """Return a dictionary with both training and validation generators."""
        # create the RAW image data generator
        x_data = CropImageDataGenerator(**self.data_gen_args)
        # create the segmentation data generator as a One-Hot tensor
        y_data = CropNumpyDataGenerator(**self.data_gen_args)
        # the dictionaries to hold generators by key value
        generators = {}
        # iterate over the subsets in the generators
        for subset in ['training', 'validation']:
            x = x_data.flow_from_directory(X_DIR, subset=subset, **self.flow_args)
            y = y_data.flow_from_directory(self.y_dir, subset=subset, **self.flow_args)
            # zip the X and y generators into a single generator
            generators[subset] = zip(x, y)

        return generators


# explicitly define the outward facing API of this module
__all__ = [CamVid.__name__]

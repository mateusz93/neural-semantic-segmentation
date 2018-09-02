"""A generator to buffer CamVid images to a Keras fit method."""
import os
from ._generators import CropImageDataGenerator
from ._generators import SegmentImageDataGenerator
from ._label_colors import SegmentationToOnehotTransformer


# the directory housing this file.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# the directory to load the X images from
X_DIR = os.path.join(THIS_DIR, 'X')
# the directory to load the y images from
Y_DIR = os.path.join(THIS_DIR, 'y')


def data_generators(
    target_size: tuple=(720, 960),
    crop_size: tuple=(224, 224),
    horizontal_flip: bool=True,
    vertical_flip: bool=True,
    validation_split: float=0.3,
    batch_size: int=3,
    shuffle: bool=True,
    seed: int=1,
):
    """
    Build a data generator for the CamVid dataset.

    Args:
        target_size: the image size of the dataset
        crop_size: the size to crop images to. if None, apply no crop
        horizontal_flip: whether to randomly flip images horizontally
        vertical_flip whether to randomly flip images vertically
        validation_split: the size of the validation set in [0, 1]
        batch_size: the number of images to load per batch
        shuffle: whether to shuffle images in the dataset
        seed: the random seed to use for the generator

    Returns:
        a dictionary of data generators with keys:
        'training': the training generator
        'validation': the validation generator

    """
    data_gen = dict(
        horizontal_flip=horizontal_flip,
        vertical_flip=vertical_flip,
        validation_split=validation_split
    )
    flow = dict(
        class_mode=None,
        target_size=target_size,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed
    )
    # create the RAW image data generator
    x_generator = CropImageDataGenerator(**data_gen, image_size=crop_size)
    # create the segmentation data generator as a One-Hot tensor
    transformer = SegmentationToOnehotTransformer()
    y_generator = SegmentImageDataGenerator(**data_gen,
        transformer=transformer,
        image_size=crop_size
    )
    # the dictionaries to hold generators by key value
    generators = {}
    # iterate over the subsets in the generators
    for subset in ['training', 'validation']:
        x = x_generator.flow_from_directory(X_DIR, subset=subset, **flow)
        y = y_generator.flow_from_directory(Y_DIR, subset=subset, **flow)
        # zip the X and y generators into a single generator
        generators[subset] = zip(x, y)

    return generators, transformer


# explicitly define the outward facing API of this module
__all__ = [data_generators.__name__]

"""A method to create a segmented version of an RGB dataset."""
import os
import glob
import shutil
import hashlib
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from ._label_colors import load_label_metadata


def _hash(mapping: dict) -> str:
    """
    Return a hash of an entire dictionary.

    Args:
        mapping: a dictionary of hash-able keys to hash-able values, i.e.,
                 __str__ should return a unique representation of each object

    Returns:
        a hash of the dictionary

    """
    # create a string to store the mapping in
    mapping_str = ''
    # iterate over the sorted dictionary keys to ensure reproducibility
    for key in sorted(mapping.keys()):
        # add the key value pairing to the string representation
        mapping_str += '{}{}'.format(key, mapping[key])
    # convert the string to bytes and return the MD5 has of the bytes
    return hashlib.md5(bytes(mapping_str, 'utf8')).hexdigest()


def create_segmented_y(
    mapping: dict=None,
    output_dtype: str='uint8',
    force_overwrite=False,
) -> pd.DataFrame:
    """
    Create a segmented version of an RGB dataset.

    Args:
        mapping: a dictionary mapping existing values to new ones for
                 dimensionality reduction
        output_dtype: the dtype of the output NumPy array of values
        force_overwrite: whether to overwrite the data if it already exists

    Returns:
        a DataFrame describing the label data mapping

    """
    # get the path to the directory with the current y data
    this_dir = os.path.dirname(os.path.abspath(__file__))
    y_dir = os.path.join(this_dir, 'y/**/**/*.png')
    # load the original label map with the mapping applied
    metadata = load_label_metadata(mapping)
    # create a vectorized method to convert RGB points to discrete codes
    codes = metadata[['rgb', 'code']].set_index('rgb')['code'].to_dict()
    codes = {(k[0] << 16) + (k[1] << 8) + k[2]: v for (k, v) in codes.items()}
    rgb_to_code = np.vectorize(codes.get, otypes=['object'])
    # get the code for the Void label to use for invalid pixels
    void_code = metadata[metadata['label'] == 'Void'].code
    # determine the number of labels and create the identity matrix
    identity = np.eye(len(metadata['label_used'].unique()))
    # create the output directory for the y data
    if mapping is None:
        new_y_dir = 'y_32'
    else:
        # use the mapping dictionary as a hash to locate its files on disk
        new_y_dir = 'y_{}'.format(_hash(mapping))
    output_dir = os.path.join(this_dir, new_y_dir)
    train = os.path.join(output_dir, 'train')
    val = os.path.join(output_dir, 'val')
    test = os.path.join(output_dir, 'test')
    # check if the metadata file exists (data is corrupt if missing)
    metadata_filename = os.path.join(output_dir, 'metadata.csv')
    if os.path.isfile(metadata_filename) and not force_overwrite:
        return output_dir
    # create all necessary directories
    for _dir in [train, val, test]:
        # delete the data directory if it exists
        shutil.rmtree(_dir, ignore_errors=True)
        # create the data directory
        _dir = os.path.join(_dir, 'data')
        os.makedirs(_dir)
    # iterate over all the files in the source directory
    for img_file in tqdm(sorted(glob.glob(y_dir)), unit='image'):
        # replace the y directory with the new directory name
        output_file = img_file.replace('/y/', '/' + new_y_dir + '/')
        # replace the file type for NumPy
        output_file = output_file.replace('.png', '.npy')
        # load the data as a NumPy array
        with Image.open(img_file) as raw_img:
            img = np.array(raw_img)
        # create a map to shift images left (to convert to hex)
        red = np.full(img.shape[:-1], 16)
        green = np.full(img.shape[:-1], 8)
        blue = np.zeros(img.shape[:-1])
        left = np.stack([red, green, blue], axis=-1).astype(int)
        # convert the image to hex and decode its discrete values
        discrete = rgb_to_code(np.left_shift(img, left).sum(axis=-1))
        # check that each pixel has been overwritten
        invalid = discrete == None
        if invalid.any():
            template = 'WARNING: {} invalid pixels in {}'
            print(template.format(invalid.sum(), img_file))
            discrete[invalid] = void_code
        # convert the discrete mapping to a one hot encoding
        onehot = identity[discrete.astype(int)].astype(output_dtype)
        # save the file to its output location
        np.save(output_file, onehot)
    # save the metadata to disk for working with the encoded data
    metadata.to_csv(metadata_filename, index=False)

    return output_dir


# explicitly define the outward facing API of this module
__all__ = [create_segmented_y.__name__]

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
    y_dir = os.path.join(this_dir, 'y/data/*.png')
    # load the original label map with the mapping applied
    label_metadata = load_label_metadata(mapping)
    # get the code for the Void label to use for invalid pixels
    void_code = label_metadata[label_metadata['label'] == 'Void'].code
    # determine the number of labels and create the identity matrix
    I = np.eye(len(label_metadata['label_used'].unique()))
    # create the output directory for the y data
    if mapping is None:
        output_dir = os.path.join(this_dir, 'y_32')
    else:
        # use the mapping dictionary as a hash to locate its files on disk
        output_dir = os.path.join(this_dir, 'y_{}'.format(_hash(mapping)))
    # check if the metadata file exists (data is corrupt if missing)
    metadata_filename = os.path.join(output_dir, 'metadata.csv')
    if os.path.isfile(metadata_filename) and not force_overwrite:
        return output_dir
    # delete the data directory if it exists
    shutil.rmtree(output_dir, ignore_errors=True)
    # create the data directory
    data_dir = os.path.join(output_dir, 'data')
    os.makedirs(data_dir)
    # iterate over all the files in the source directory
    for img_file in tqdm(sorted(glob.glob(y_dir))):
        # get the name of the output file
        output_file = os.path.basename(os.path.normpath(img_file))
        # replace png extension with npy for NumPy file
        output_file = output_file.replace('.png', '.npy')
        # add the file name to the data directory path to save to
        output_file = os.path.join(data_dir, output_file)
        # load the data as a NumPy array
        with Image.open(img_file) as raw_img:
            img = np.array(raw_img)
        # create a placeholder for the new image's discrete coding
        discrete = np.empty(img.shape[:-1])
        discrete[:] = -1
        # iterate over the RGB points in the dataset
        for rgb in label_metadata['rgb']:
            # extract the discrete code for this RGB point
            code = label_metadata[label_metadata['rgb'] == rgb].code
            # set all points equal to this in the image to the discrete code
            discrete[(img == rgb).all(axis=-1)] = code
        # check that each pixel has been overwritten
        if (discrete == -1).sum() > 0:
            invalid = (discrete == -1).sum()
            print('WARNING: {} invalid pixels in {}'.format(invalid, img_file))
            discrete[discrete == -1] = void_code
        # convert the discrete mapping to a one hot encoding
        onehot = I[discrete.astype(int)].astype(output_dtype)
        # save the file to its output location
        np.save(output_file, onehot)
    # save the metadata to disk for working with the encoded data
    label_metadata.to_csv(metadata_filename, index=False)

    return output_dir


# explicitly define the outward facing API of this module
__all__ = [create_segmented_y.__name__]

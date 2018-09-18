"""A method to create a segmented version of an RGB dataset."""
import os
import glob
import shutil
from PIL import Image
import numpy as np
from tqdm import tqdm
from ._label_colors import load_label_metadata


def create_segmented_y(
    mapping: dict=None,
    output_dtype: str='uint8',
    force_overwrite=False,
) -> tuple:
    """
    Create a segmented version of an RGB dataset.

    Args:
        mapping: a dictionary mapping existing values to new ones for
                 dimensionality reduction
        output_dtype: the dtype of the output numpy array of values
        force_overwrite: whether to overwrite the data if it already exists

    Returns:
        a tuple of:
        - the directory the label data was saved to
        - a dataframe describing the label data mapping

    """
    # get the path to the directory with the current y data
    this_dir = os.path.dirname(os.path.abspath(__file__))
    y_dir = os.path.join(this_dir, 'y/data/*.png')
    # load the original label map
    label_metadata = load_label_metadata(mapping)
    void_code = label_metadata[label_metadata['label'] == 'Void'].code
    # determine the number of labels and create the identity matrix
    num_labels = len(label_metadata['label_used'].unique())
    # create the output directory for the data
    output_dir = os.path.join(this_dir, 'y_{}'.format(num_labels))
    data_dir = os.path.join(output_dir, 'data'.format(num_labels))
    # check if the directory exists and return early if force overwrite
    # is disabled
    if os.path.isdir(data_dir):
        if not force_overwrite:
            return output_dir, label_metadata
    # delete the directory if it exists
    shutil.rmtree(data_dir, ignore_errors=True)
    # create the directory
    os.makedirs(data_dir)
    # iterate over all the files in the source directory
    for img_file in tqdm(sorted(glob.glob(y_dir))):
        # get the name of the output file
        output_file = os.path.basename(os.path.normpath(img_file))
        output_file = output_file.replace('.png', '.npy')
        output_file = os.path.join(data_dir, output_file)
        # load the image
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
            print('WARNING invalid pixels in: {}'.format(img_file))
            discrete[discrete == -1] = void_code
        # convert the discrete mapping to a one hot encoding
        onehot = np.eye(num_labels)[discrete.astype(int)].astype(output_dtype)
        # save the file to its output location
        np.save(output_file, onehot)
    # save the metadata to disk for working with the encoded data
    metadata_filename = os.path.join(output_dir, 'metadata.csv')
    label_metadata.to_csv(metadata_filename, index=False)

    return output_dir, label_metadata


# explicitly define the outward facing API of this module
__all__ = [create_segmented_y.__name__]

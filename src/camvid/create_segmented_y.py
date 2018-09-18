"""A method to create a segmented version of an RGB dataset."""
import os
import glob
import shutil
from PIL import Image
import numpy as np
from ._label_colors import load_label_metadata


def create_segmented_y(mapping: dict=None):
    """
    Create a segmented version of an RGB dataset.

    Args:

    Returns:
        None

    """
    # get the path to the directory with the current y data
    this_dir = os.path.dirname(os.path.abspath(__file__))
    y_dir = os.path.join(this_dir, 'y/data/*.png')
    # load the original label map
    label_metadata = load_label_metadata(mapping)
    # determine the number of labels and create the identity matrix
    num_labels = len(label_metadata['label_used'].unique())
    I = np.eye(num_labels)
    # create the output directory for the data
    output_dir = os.path.join(this_dir, 'y_{}/data'.format(num_labels))
    # delete the directory if it exists
    shutil.rmtree(output_dir, ignore_errors=True)
    # create the directory
    os.makedirs(output_dir)
    # iterate over all the files in the source directory
    for y in sorted(glob.glob(y_dir)):
        # get the name of the output file
        output_file = os.path.basename(os.path.normpath(y))
        output_file = output_file.replace('.png', '.npy')
        output_file = os.path.join(output_dir, output_file)
        # load the image
        img = np.array(Image.open(y))
        # create a placeholder for the new image's discrete coding
        discrete = np.empty(img.shape[:-1])
        # iterate over the RGB points in the dataset
        for rgb in label_metadata['rgb']:
            # extract the discrete code for this RGB point
            code = label_metadata[label_metadata['rgb'] == rgb].code
            # set all points equal to this in the image to the discrete code
            discrete[(img == rgb).all(axis=-1)] = code
        # convert the discrete mapping to a one hot encoding
        onehot = I[discrete.astype(int)]
        # save the file to its output location
        np.save(output_file, onehot)

    print(label_metadata)


# explicitly define the outward facing API of this module
__all__ = [create_segmented_y.__name__]

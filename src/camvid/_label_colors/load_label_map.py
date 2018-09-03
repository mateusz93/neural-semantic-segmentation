"""A Method to load the label map data from disk."""
import os
import pandas as pd


def load_label_map():
    """Return the label map based on RGB encoding."""
    # the path to this directory
    this_dir = os.path.dirname(os.path.abspath(__file__))
    map_file = os.path.join(this_dir, 'label_colors.txt')
    # the names of the columns
    names = ['R', 'G', 'B', 'label']
    # load the table from the file
    label_map = pd.read_table(map_file, sep=r'\s+', names=names)
    # create the discrete encoding for each label
    label_map['encoding'] = label_map.index
    # set the index to the tuple of RGB
    label_map.index = list(zip(label_map.R, label_map.G, label_map.B))
    label_map.index.name = 'rgb'
    # remove the individual RGB columns
    del label_map['R']
    del label_map['G']
    del label_map['B']

    return {
        'encoding': label_map.to_dict()['encoding'],
        'labels': label_map.set_index('encoding').to_dict()['label'],
    }


# explicitly define the outward facing API of this module
__all__ = [load_label_map.__name__]

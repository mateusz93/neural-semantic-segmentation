"""A Method to load the label map data from disk."""
import os
import pandas as pd


def load_label_metadata(mapping: dict=None) -> pd.DataFrame:
    """
    Return the data frame mapping RGB points to string label data.

    Args:
        mapping: a dictionary of replacement values for labels

    Returns:
        a pandas DataFrame with the label metadata

    """
    # the path to this directory
    this_dir = os.path.dirname(os.path.abspath(__file__))
    map_file = os.path.join(this_dir, 'label_colors.txt')
    # the names of the columns
    names = ['R', 'G', 'B', 'label']
    # load the table from the file
    label_map = pd.read_table(map_file, sep=r'\s+', names=names)
    # set the index to the tuple of RGB
    label_map['rgb'] = list(zip(label_map.R, label_map.G, label_map.B))
    # remove the individual RGB columns
    del label_map['R']
    del label_map['G']
    del label_map['B']

    # apply the mapping if provided
    if mapping is not None:
        # apply the mapping if specified to generate the new labels
        label_map['label_used'] = label_map['label'].replace(mapping)
        # get the draw value for the new labels based on original colors
        label_map['rgb_draw'] = label_map.set_index('label').loc[label_map['label_used']]['rgb'].values
    else:
        # use the data labels
        label_map['label_used'] = label_map['label']
        # use the data RGB points
        label_map['rgb_draw'] = label_map['rgb']

    # convert the used labels to a categorical variable of integers
    label_map['code'] = label_map['label_used'].astype('category').cat.codes

    return label_map


# explicitly define the outward facing API of this module
__all__ = [load_label_metadata.__name__]

"""A method to load a class mapping from disk."""
import os
import pandas as pd


# get a handle to the absolute path of this directory
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# get the default file for a mapping to load
DEFAULT_FILE = os.path.join(THIS_DIR, '11_class.txt')


def load_mapping(mapping_file=DEFAULT_FILE, sep=r'\s+'):
    """
    Load a mapping file from disk as a dictionary.

    Args:
        mapping_file: file like object pointing to a text file with mapping data
        sep: the separator for entries in the file

    Returns:
        a dictionary mapping old classes to generalized classes

    """
    # the names of the columns in the file
    names = ['og', 'new']
    # load the DataFrame with the original classes as the index col
    mapping = pd.read_table(mapping_file, sep=sep, names=names, index_col='og')
    # return a dictionary of the new column mapping old classes to new classes
    return mapping['new'].to_dict()


# explicitly define the outward facing API of this module
__all__ = [load_mapping.__name__]

"""A method to convert Keras fit history to a pandas DataFrame of results."""
import pandas as pd


def history_to_results(history):
    """
    Convert a Keras history object into results.

    Args:
        history: the history object to extract training history from

    Returns:
        a data frame with the training and validation data

    """
    # make a data-frame from the fit history
    history = pd.DataFrame(history.history)
    # set the index name to 'Epoch'
    history.index.name = 'Epoch'
    # extract the training data
    train = history[[c for c in history.columns if 'val_' not in c]]
    train.columns = train.columns.str.replace('iou_', '')
    # extract the validation data
    val = history[[c for c in history.columns if 'val_' in c]]
    val.columns = val.columns.str.replace('val_', '')
    val.columns = val.columns.str.replace('iou_', '')
    rows = [train.iloc[-1], val.iloc[-1]]
    # create the data-frame over the rows and transpose the index to columns
    return pd.DataFrame(rows, index=['train', 'val']).T


# explicitly define the outward facing API of this module
__all__ = [history_to_results.__name__]

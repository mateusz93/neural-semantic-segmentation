"""A metric to calculate categorical accuracy."""
import numpy as np
from sklearn.metrics import confusion_matrix


def categorical_accuracy(y_true: np.ndarray, y_pred: np.ndarray,
    mask: np.ndarray=None
) -> float:
    """
    Return a categorical accuracy tensor for label and prediction tensors.

    Args:
        y_true: the ground truth labels to compare against
        y_pred: the predicted labels from a loss network
        mask: the mask to use for the metric

    Returns:
        a tensor of the categorical accuracy between truth and predictions

    """
    # get number of labels to calculate IoU for
    num_classes = y_pred.shape[-1]
    # set the mask to all 1 if there are none specified
    _weights = np.ones(num_classes) if mask is None else mask
    # extract the label using ArgMax and flatten into a 1D vector
    y_true = np.argmax(y_true, axis=-1).flatten()
    y_pred = np.argmax(y_pred, axis=-1).flatten()
    # calculate the confusion matrix of the ground truth and predictions
    confusion = confusion_matrix(y_true, y_pred, list(range(num_classes)))
    # extract the number of correct guesses from the diagonal
    correct = _weights * np.sum(confusion * np.eye(num_classes), axis=-1)
    # extract the number of total values per class from ground truth
    total = _weights * np.sum(confusion, axis=-1)
    # calculate the total accuracy
    return np.sum(correct) / np.sum(total)


# explicitly define the outward facing API of this module
__all__ = [categorical_accuracy.__name__]

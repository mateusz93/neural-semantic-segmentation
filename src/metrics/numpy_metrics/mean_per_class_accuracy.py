"""A metric to calculate per class accuracy."""
import numpy as np
from sklearn.metrics import confusion_matrix


def mean_per_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray,
    mask: np.ndarray=None
) -> float:
    """
    Return the per class accuracy for ground truth and prediction tensors.

    Args:
        y_true: the expected y values as a one-hot
        y_pred: the predicted y values as a one-hot or probability tensor
        mask: the mask to use for the metric

    Returns:
        the per class accuracy of y_pred in relation to y_true

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
    correct = np.sum(confusion * np.eye(num_classes), axis=-1)
    # extract the number of total values per class from ground truth
    total = np.sum(confusion, axis=-1)
    # get per class accuracy by dividing correct by total. use epsilon to
    # prevent divide by zero (when total is 0 for label, correct is 0 too)
    per_class_acc = _weights * correct / (total + np.finfo(np.float32).eps)
    # calculate the mean per class accuracy
    return np.sum(per_class_acc, axis=-1) / _weights.sum()


# explicitly define the outward facing API of this module
__all__ = [mean_per_class_accuracy.__name__]

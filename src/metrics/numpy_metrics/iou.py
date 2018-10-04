"""An implementation of the Intersection over Union (IoU) metric for Keras."""
import numpy as np


def iou(y_true: np.ndarray, y_pred: np.ndarray, label: int) -> float:
    """
    Return the Intersection over Union (IoU) for a given label.

    Args:
        y_true: the expected y values as a one-hot
        y_pred: the predicted y values as a one-hot or probability tensor
        c: the label to return the IoU for

    Returns:
        the IoU for the given label

    Reference:
        https://en.wikipedia.org/wiki/Jaccard_index

    """
    # extract the label values using the ArgMax operator then
    # calculate equality of the predictions and truths to the label
    y_true = np.argmax(y_true, axis=-1) == label
    y_pred = np.argmax(y_pred, axis=-1) == label
    # calculate the |intersection| (AND) of the labels
    intersection = np.sum(y_true * y_pred)
    # calculate the |union| (OR) of the labels
    union = np.sum(y_true) + np.sum(y_pred) - intersection
    # calculate the intersection over union
    return intersection / union


def mean_iou(y_true: np.ndarray, y_pred: np.ndarray,
    mask: np.ndarray=None
) -> float:
    """
    Return the mean Intersection over Union (IoU) of truths and predictions.

    Args:
        y_true: the expected y values as a one-hot
        y_pred: the predicted y values as a one-hot or probability tensor
        mask: the weights to use for the metric

    Returns:
        the scalar IoU value (mean over all labels)

    Reference:
        https://en.wikipedia.org/wiki/Weighted_arithmetic_mean

    """
    # get number of labels to calculate IoU for
    num_classes = y_pred.shape[-1]
    # set the mask to all 1 if there are none specified
    _weights = np.ones(num_classes) if mask is None else mask
    # initialize a variable to store total IoU in
    total_iou = 0
    # iterate over labels to calculate IoU for
    for label in range(num_classes):
        total_iou = total_iou + _weights[label] * iou(y_true, y_pred, label)
    # divide total IoU by number of labels to get mean IoU
    return total_iou / _weights.sum()


# explicitly define the outward facing API of this module
__all__ = [mean_iou.__name__, iou.__name__]

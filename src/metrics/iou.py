"""An implementation of the Intersection over Union (IoU) metric for Keras."""
import numpy as np
from keras import backend as K


def iou(y_true, y_pred, label: int):
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
    y_true = K.cast(K.equal(K.argmax(y_true), label), K.floatx())
    y_pred = K.cast(K.equal(K.argmax(y_pred), label), K.floatx())
    # calculate the |intersection| (AND) of the labels
    intersection = K.sum(y_true * y_pred)
    # calculate the |union| (OR) of the labels
    union = K.sum(y_true) + K.sum(y_pred) - intersection
    # avoid divide by zero - if the union is zero, return 1
    # otherwise, return the intersection over union
    return K.switch(K.equal(union, 0), 1.0, intersection / union)


def build_iou_for(label: int, name: str=None):
    """
    Build an Intersection over Union (IoU) metric for a label.

    Args:
        label: the label to build the IoU metric for
        name: an optional name for debugging the built method

    Returns:
        a Keras metric to evaluate IoU for the given label

    Note:
        label and name support list inputs for multiple labels

    """
    # handle recursive inputs (e.g. a list of labels and names)
    if isinstance(label, list):
        if isinstance(name, dict):
            return [build_iou_for(l, name[l]) for l in label]
        return [build_iou_for(l) for l in label]

    # build the method for returning the IoU of the given label
    def label_iou(y_true, y_pred):
        """
        Return the Intersection over Union (IoU) score for {0}.

        Args:
            y_true: the expected y values as a one-hot
            y_pred: the predicted y values as a one-hot or probability tensor

        Returns:
            the scalar IoU value for the given label ({0})

        """.format(label)
        return iou(y_true, y_pred, label)

    # if no name is provided, use the label
    if name is None:
        name = label
    # change the name of the method for debugging
    label_iou.__name__ = 'iou_{}'.format(name)

    return label_iou


def build_mean_iou(weights=None):
    """
    Build a mean intersection over union metric.

    Args:
        weights: the weights to use for the metric

    Returns:
        a callable mean intersection over union metric

    """
    def mean_iou(y_true, y_pred):
        """
        Return the Intersection over Union (IoU) score.

        Args:
            y_true: the expected y values as a one-hot
            y_pred: the predicted y values as a one-hot or probability tensor

        Returns:
            the scalar IoU value (mean over all labels)

        Reference:
            https://en.wikipedia.org/wiki/Weighted_arithmetic_mean

        """
        # get number of labels to calculate IoU for
        num_classes = K.int_shape(y_pred)[-1]
        # set the weights to all 1 if there are none specified
        _weights = np.ones(num_classes) if weights is None else weights
        # initialize a variable to store total IoU in
        total_iou = K.variable(0)
        # iterate over labels to calculate IoU for
        for label in range(num_classes):
            total_iou = total_iou + _weights[label] * iou(y_true, y_pred, label)
        # divide total IoU by number of labels to get mean IoU
        return total_iou / _weights.sum()

    return mean_iou


# explicitly define the outward facing API of this module
__all__ = [
    build_iou_for.__name__,
    build_mean_iou.__name__
]

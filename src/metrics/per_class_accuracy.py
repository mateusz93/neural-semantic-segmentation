"""A Keras metric to calculate the per class accuracy in categorical models."""
from keras import backend as K


def class_accuracy(y_true, y_pred, label: int):
    """
    Return the accuracy for a given label.

    Args:
        y_true: the expected y values as a one-hot
        y_pred: the predicted y values as a one-hot or Softmax output
        c: the label to return the accuracy for

    Returns:
        the accuracy for the given label

    """
    # extract the label values using the ArgMax operator then calculate
    # equality of the predictions and truths to the label
    y_true = K.equal(K.argmax(y_true), label)
    y_pred = K.equal(K.argmax(y_pred), label)
    # calculate the accuracy as average of correct predictions
    return K.mean(K.cast(K.equal(y_true, y_pred), K.floatx()))


def per_class_accuracy(y_true, y_pred):
    """
    Return the per class accuracy of predictions.

    Args:
        y_true: the ground truth labels
        y_pred: the predicted labels

    Returns:
        the per class accuracy of y_pred based on y_true

    """
    # get number of labels to calculate class accuracy for
    num_labels = K.int_shape(y_pred)[-1]
    # initialize a variable to store total accuracy in
    acc = K.variable(0)
    # iterate over labels to calculate accuracy for
    for label in range(num_labels):
        acc = acc + class_accuracy(y_true, y_pred, label)
    # divide total acc by number of labels to get per class accuracy
    return acc / num_labels


# explicitly define the outward facing API of this module
__all__ = [per_class_accuracy.__name__]

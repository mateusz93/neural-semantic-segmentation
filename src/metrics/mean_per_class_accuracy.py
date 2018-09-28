"""A metric to calculate per class accuracy."""
from keras import backend as K
from ..backend.tensorflow_backend import confusion_matrix


def build_mean_per_class_accuracy(weights=None):
    """
    Build a mean per class accuracy method using given weights.

    Args:
        weights: the weights to use for the metric

    Returns:
        a callable mean per class accuracy evaluation metric

    """
    def mean_per_class_accuracy(y_true, y_pred):
        """
        Return the per class accuracy for ground truth and prediction tensors.

        Args:
            y_true: the expected y values as a one-hot
            y_pred: the predicted y values as a one-hot or probability tensor

        Returns:
            the per class accuracy of y_pred in relation to y_true

        """
        # get number of labels to calculate IoU for
        num_classes = K.int_shape(y_pred)[-1]
        # set the weights to all 1 if there are none specified
        _weights = np.ones(num_classes) if weights is None else weights
        # extract the label using ArgMax and flatten into a 1D vector
        y_true = K.flatten(K.argmax(y_true, axis=-1))
        y_pred = K.flatten(K.argmax(y_pred, axis=-1))
        # calculate the confusion matrix of the ground truth and predictions
        confusion = confusion_matrix(y_true, y_pred, num_classes=num_classes)
        # confusion will return integers, but we need floats to multiply by eye
        confusion = K.cast(confusion, K.floatx())
        # extract the number of correct guesses from the diagonal
        correct = K.sum(confusion * K.eye(num_classes), axis=-1)
        # extract the number of total values per class from ground truth
        total = K.sum(confusion, axis=-1)
        # get per class accuracy by dividing correct by total. use epsilon to
        # prevent divide by zero (when total is 0 for label, correct is 0 too)
        per_class_acc = _weights * correct / (total + K.epsilon())
        # calculate the mean per class accuracy
        return K.sum(per_class_acc, axis=-1) / _weights.sum()

    return mean_per_class_accuracy


# explicitly define the outward facing API of this module
__all__ = [build_mean_per_class_accuracy.__name__]

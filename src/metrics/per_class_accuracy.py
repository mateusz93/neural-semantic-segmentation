"""A Keras metric to calculate the per class accuracy given some weights."""
from keras import backend as K

# weights info: https://github.com/alexgkendall/caffe-segnet/issues/37
def build_per_class_accuracy(weights=None):
    """
    Build a per class metric accuracy with input parameters.

    Args:
        weights: weights matching rank with predicted labels

    Returns:
        a callable Keras metric for per class accuracy

    """
    if weights is not None:
        if K.int_shape(weights) > 1:
            weights = K.reshape(weights, -1)
    # build the metric with the given parameters
    def per_class_accuracy(y_true, y_pred):
        """
        Return the per class accuracy of predictions.

        Args:
            y_true: the ground truth labels
            y_pred: the predicted labels

        Returns:
            the per class accuracy of y_pred based on y_true

        """
        # y_true = K.flatten(K.argmax(y_true, axis=-1))
        # y_pred = K.flatten(K.argmax(y_pred, axis=-1))
        # is_correct = K.cast(K.equal(y_true, y_pred), K.floatx())

        # if weights is not None:
        #     is_correct *= weights[y_true]

        # return K.mean(is_correct)

    return per_class_accuracy


# explicitly define the outward facing API of this module
__all__ = [build_per_class_accuracy.__name__]

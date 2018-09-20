"""A Keras implementation of weighted categorical cross entropy loss."""
from keras import backend as K
from keras.losses import categorical_crossentropy


def build_weighted_categorical_crossentropy(weights=None):
    """
    Build a weighted categorical crossentropy loss function.

    Args:
        weights: the weights to use for the loss function
    """
    # build the weighted categorical crossentropy method
    def weighted_categorical_crossentropy(y_true, y_pred):
        """
        Return the weighted categorical crossentropy.

        Args:
            y_true: the ground truth labels
            y_pred: the predicted labels from a network

        Returns:
            a symbolic tensor for the weighted crossentropy

        """
        if weights is None:
            return categorical_crossentropy(y_true, y_pred)
        # scale preds so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # manual computation of crossentropy
        y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())
        loss = y_true * K.log(y_pred) * weights
        loss = - K.sum(loss, axis=-1)

        return loss

    return weighted_categorical_crossentropy


# explicitly define the outward facing API of this module
__all__ = [build_weighted_categorical_crossentropy.__name__]

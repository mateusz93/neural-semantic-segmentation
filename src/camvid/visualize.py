"""Methods to visualize data from the dataset."""
import numpy as np
from matplotlib import pyplot as plt


def plot(
    X: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray=None,
) -> None:
    """
    Plot the original image, the true y, and an optional predicted y.

    Args:
        X: the original image
        y_true: the actual segmentation mapping
        y_pred: the predicted segmentation mapping

    Returns:
        None

    """
    # setup the values to display
    if y_pred is None:
        values = (X, y_true)
        title = '$X, y$'
    else:
        values = (X, y_true, y_pred)
        title = '$X, y, \\hat{y}$'
    # concatenate the values to display into a single tensor
    img = np.concatenate(values, axis=1).astype('uint8') / 255
    # plot the values
    plt.imshow(img)
    plt.title(title)


# explicitly define the outward facing API of this module
__all__ = [plot.__name__]

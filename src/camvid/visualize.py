"""Methods to visualize data from the dataset."""
from matplotlib import pyplot as plt


def plot(**kwargs: dict) -> None:
    """
    Plot the original image, the true y, and an optional predicted y.

    Args:
        kwargs: images to plot

    Returns:
        None

    """
    # create subplots for each image
    _, axarr = plt.subplots(1, len(kwargs))
    # iterate over the images in the dictionary
    for idx, (title, img) in enumerate(kwargs.items()):
        # plot the image
        axarr[idx].imshow(img.astype('uint8') / 255)
        # set the title for this subplot
        axarr[idx].set_title(title)
        # remove the ticks from the x and y axes
        axarr[idx].xaxis.set_major_locator(plt.NullLocator())
        axarr[idx].yaxis.set_major_locator(plt.NullLocator())


# explicitly define the outward facing API of this module
__all__ = [plot.__name__]

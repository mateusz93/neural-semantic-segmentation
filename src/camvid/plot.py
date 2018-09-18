"""Methods to visualize data from the dataset."""
from matplotlib import pyplot as plt


# the DPI for CamVid images is 72.0
DPI = 36.0


def plot(dpi: float=96.0, **kwargs: dict) -> None:
    """
    Plot the original image, the true y, and an optional predicted y.

    Args:
        dpi: the DPI of the figure to render
        kwargs: images to plot

    Returns:
        None

    """
    # determine the image shape for platting based on DPI
    image_shape = list(kwargs.values())[0].shape
    figsize = image_shape[0] / DPI, image_shape[1] / DPI
    # create subplots for each image
    _, axarr = plt.subplots(len(kwargs), 1, figsize=figsize, dpi=dpi)
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

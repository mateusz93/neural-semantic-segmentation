"""A method to map vectors to color maps."""
from matplotlib import pyplot as plt


def heatmap(arr):
    """
    Use the CMRmap color map to convert the input array to a heat-map.

    Args:
        arr: the array to convert to a heat-map

    Returns:
        arr converted to a heat-map via the CMRmap color map

    """
    # normalize the input data
    arr = plt.Normalize()(arr)
    # return the color map with the alpha channel omitted
    heat = plt.cm.jet(arr)[..., :-1] * 255

    return heat.astype('uint8')


# explicitly define the outward facing API of this module
__all__ = [heatmap.__name__]

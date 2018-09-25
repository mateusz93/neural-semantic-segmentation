"""A method to map vectors to color maps."""
from matplotlib import pyplot as plt


def heatmap(arr, color_map='cubehelix'):
    """
    Use the CMRmap color map to convert the input array to a heat-map.

    Args:
        arr: the array to convert to an RGB heat-map
        color_map: the color map to use (defaults to 'cubehelix')
        -   cubehelix is like jet, but with a better luminosity gradient to
            illustrate the scale to the human eye, as well as support black
            and white printing (i.e., black and white cube helix will resemble
            plt.cm.binary or plt.cm.Greys)

    Returns:
        arr converted to a heat-map via the CMRmap color map

    """
    # normalize the input data
    arr = plt.Normalize()(arr)
    # return the color map with the alpha channel omitted
    heat = getattr(plt.cm, color_map)(arr)[..., :-1] * 255

    return heat.astype('uint8')


# explicitly define the outward facing API of this module
__all__ = [heatmap.__name__]

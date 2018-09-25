"""A method to map vectors to color maps."""
import numpy as np
from matplotlib import pyplot as plt


def heatmap(arr: np.ndarray, color_map='cubehelix') -> np.ndarray:
    """
    Use the given color map to convert the input vector to a heat-map.

    Args:
        arr: the vector to convert to an RGB heat-map
        color_map: the color map to use (defaults to 'cubehelix')
        -   cubehelix is like jet, but with a better luminosity gradient to
            illustrate the scale to the human eye, as well as support black
            and white printing (i.e., black and white cube helix will resemble
            plt.cm.binary or plt.cm.Greys)

    Returns:
        arr mapped to RGB using the given color map (vector of bytes)

    """
    # normalize the input data
    arr = plt.Normalize()(arr)
    # unwrap the color map from matplotlib
    color_map = plt.cm.get_cmap(color_map)
    # get the heat-map from the color map in RGB (i.e., omit the alpha channel)
    _heatmap = color_map(arr)[..., :-1]
    # scale heat-map from [0,1] to [0, 255] as a vector of bytes
    _heatmap = (255 * _heatmap).astype(np.uint8)

    return _heatmap


# explicitly define the outward facing API of this module
__all__ = [heatmap.__name__]

"""A method to repeat a generators output for multi-IO models."""


def repeat_generator(x, y, x_repeats=0, y_repeats=0):
    """
    Return a generator that repeats x and y input generators.

    Args:
        x: a generator for x data
        y: a generator for y data
        x_repeats: the number of times to repeat the x data (default 0)
        y_repeats: the number of times to repeat the y data (default 0)

    Returns:
        a new generator that returns a tuple of lists size (x_repeats + 1) and
        (y_repeats + 1) respectively

    """
    # create a mapping function to repeat the x and y inputs
    def repeat_outputs(_x, _y):
        # repeat if there is a positive repeat rate
        if x_repeats > 0:
            _x = [_x] * (x_repeats + 1)
        # repeat if there is a positive repeat rate
        if y_repeats > 0:
            _y = [_y] * (y_repeats + 1)
        # return the updated x and y values
        return _x, _y
    # return the new mapping for the generators
    return map(repeat_outputs, x, y)


# explicitly define the outward facing API of this module
__all__ = [repeat_generator.__name__]

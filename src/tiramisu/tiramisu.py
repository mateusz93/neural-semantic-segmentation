"""The standard Tiramisu model."""
from keras.layers import Activation
from keras.models import Model
from keras.optimizers import RMSprop
from ..losses import build_categorical_crossentropy
from ..metrics import build_categorical_accuracy
from ._core import build_tiramisu


def tiramisu(image_shape: tuple, num_classes: int,
    class_weights=None,
    initial_filters: int=48,
    growth_rate: int=16,
    layer_sizes: list=[4, 5, 7, 10, 12],
    bottleneck_size: int=15,
    dropout: float=0.2,
    learning_rate: float=1e-3,
) -> Model:
    """
    Build a Tiramisu model for the given image shape.

    Args:
        image_shape: the image shape to create the model for
        num_classes: the number of classes to segment for (e.g. c)
        class_weights: the weights for each class
        initial_filters: the number of filters in the first convolution layer
        growth_rate: the growth rate to use for the network (e.g. k)
        layer_sizes: a list with the size of each dense down-sample block.
                     reversed to determine the size of the up-sample blocks
        bottleneck_size: the number of convolutional layers in the bottleneck
        dropout: the dropout rate to use in dropout layers
        learning_rate: the learning rate for the RMSprop optimizer

    Returns:
        a compiled model of the Tiramisu architecture

    """
    # build the base of the network
    inputs, logits = build_tiramisu(image_shape, num_classes,
        initial_filters=initial_filters,
        growth_rate=growth_rate,
        layer_sizes=layer_sizes,
        bottleneck_size=bottleneck_size,
        dropout=dropout,
    )
    # pass the logits through the Softmax activation to get probabilities
    softmax = Activation('softmax', name='softmax')(logits)
    # build the Tiramisu model
    model = Model(inputs=[inputs], outputs=[softmax])
    # compile the model
    model.compile(
        optimizer=RMSprop(lr=learning_rate),
        loss=build_categorical_crossentropy(class_weights),
        metrics=[build_categorical_accuracy(weights=class_weights)],
    )

    return model


def predict(model, generator, camvid):
    """
    Return post-processed predictions for the given generator.

    Args:
        model: the tiramisu model to use to predict with
        generator: the generator to get data from
        camvid: the camvid instance for unmapping target values

    Returns:
        a tuple of for NumPy tensors with RGB data:
        - the batch of RGB X values
        - the unmapped RGB batch of y values
        - the unmapped RGB predicted values from the model

    """
    # generate a batch of data from the generator
    imgs, y_true = next(generator)
    # get predictions from the model
    y_pred = model.predict(imgs)
    # return a tuple of RGB pixel data
    return imgs, camvid.unmap(y_true), camvid.unmap(y_pred)


# explicitly define the outward facing API of this module
__all__ = [tiramisu.__name__, predict.__name__]

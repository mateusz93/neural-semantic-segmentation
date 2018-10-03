from keras.layers import Activation
from keras.models import Model
from keras.optimizers import RMSprop
from ..layers import Stack
from ..losses import build_categorical_crossentropy
from ..losses import build_categorical_aleatoric_loss
from ..metrics import build_categorical_accuracy
from ..utils import extract_aleatoric
from ..utils import heatmap
from ._core import build_tiramisu


def aleatoric_tiramisu(image_shape: tuple, num_classes: int,
    class_weights=None,
    initial_filters: int=48,
    growth_rate: int=16,
    layer_sizes: list=[4, 5, 7, 10, 12],
    bottleneck_size: int=15,
    dropout: float=0.2,
    learning_rate: float=1e-3,
    samples: int=50,
):
    """
    Build a Tiramisu model that computes Aleatoric uncertainty.

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
        samples: the number of samples for Monte Carlo loss estimation

    Returns:
        a compiled model of the Tiramisu architecture + Aleatoric

    """
    # build the base of the network
    inputs, logits, sigma = build_tiramisu(image_shape, num_classes,
        initial_filters=initial_filters,
        growth_rate=growth_rate,
        layer_sizes=layer_sizes,
        bottleneck_size=bottleneck_size,
        dropout=dropout,
        split_head=True,
    )
    # stack the logits and sigma for aleatoric loss
    aleatoric = Stack(name='aleatoric')([logits, sigma])
    # pass the logits through the Softmax activation to get probabilities
    softmax = Activation('softmax', name='softmax')(logits)
    # build the Tiramisu model
    tiramisu = Model(inputs=[inputs], outputs=[softmax, sigma, aleatoric])

    # compile the model
    tiramisu.compile(
        optimizer=RMSprop(lr=learning_rate),
        loss={
            'softmax': build_categorical_crossentropy(class_weights),
            'aleatoric': build_categorical_aleatoric_loss(samples)
        },
        metrics={'softmax': [build_categorical_accuracy(weights=class_weights)]},
    )

    return tiramisu


def predict(model, generator, camvid) -> tuple:
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
        - the unmapped RGB predicted mean values from the model
        - the meatmap RGB values of the model variance

    """
    # get the batch of data
    imgs, y_true = next(generator)
    # predict mean values and variance
    mean, sigma2, _ = model.predict(imgs)
    # extract the aleatoric uncertainty from the tensor
    sigma2 = extract_aleatoric(sigma2, mean)
    # return X values, unmapped y and u values, and heatmap of s2
    return (
        imgs,
        camvid.unmap(y_true[0]),
        camvid.unmap(mean),
        heatmap(sigma2, 'afmhot'),
    )


# explicitly define the outward facing API of this module
__all__ = [aleatoric_tiramisu.__name__, predict.__name__]

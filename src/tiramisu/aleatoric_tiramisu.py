"""Tiramisu model that has head split to estimate aleatoric uncertainty."""
from keras.layers import Activation
from keras.layers import Input
from keras.models import Model
from keras.optimizers import RMSprop
from ..layers import Stack
from ..losses import build_categorical_crossentropy
from ..losses import build_categorical_aleatoric_loss
from ..metrics import build_categorical_accuracy
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
    weights_file: str=None,
    freeze_weights: bool=True,
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
    # pass the logits through the Softmax activation to get probabilities
    softmax = Activation('softmax', name='softmax')(logits)
    # build the Tiramisu model
    tiramisu = Model(inputs=[inputs], outputs=[softmax])
    tiramisu.load_weights(weights_file)
    tiramisu.trainable = False

    # stack the logits and sigma for aleatoric loss
    aleatoric = Stack(name='aleatoric')([logits, sigma])
    # pass the logits through the Softmax activation to get probabilities
    softmax = Activation('softmax', name='softmax')(logits)
    # build the Tiramisu model
    model = Model(inputs=[inputs], outputs=[softmax, sigma, aleatoric])

    # compile the model
    model.compile(
        optimizer=RMSprop(lr=learning_rate),
        loss={
            'softmax': build_categorical_crossentropy(class_weights),
            'aleatoric': build_categorical_aleatoric_loss(samples)
        },
        metrics={'softmax': [build_categorical_accuracy(weights=class_weights)]},
    )

    return model


# def aleatoric_tiramisu(image_shape: tuple, num_classes: int,
#     class_weights=None,
#     initial_filters: int=48,
#     growth_rate: int=16,
#     layer_sizes: list=[4, 5, 7, 10, 12],
#     bottleneck_size: int=15,
#     dropout: float=0.2,
#     learning_rate: float=1e-3,
#     samples: int=50,
#     weights_file: str=None,
#     freeze_weights: bool=True,
# ):
#     """
#     Build a Tiramisu model that computes Aleatoric uncertainty.

#     Args:
#         image_shape: the image shape to create the model for
#         num_classes: the number of classes to segment for (e.g. c)
#         class_weights: the weights for each class
#         initial_filters: the number of filters in the first convolution layer
#         growth_rate: the growth rate to use for the network (e.g. k)
#         layer_sizes: a list with the size of each dense down-sample block.
#                      reversed to determine the size of the up-sample blocks
#         bottleneck_size: the number of convolutional layers in the bottleneck
#         dropout: the dropout rate to use in dropout layers
#         learning_rate: the learning rate for the RMSprop optimizer
#         samples: the number of samples for Monte Carlo loss estimation

#     Returns:
#         a compiled model of the Tiramisu architecture + Aleatoric

#     """
#     # build the base of the network
#     inputs, logits, sigma = build_tiramisu(image_shape, num_classes,
#         initial_filters=initial_filters,
#         growth_rate=growth_rate,
#         layer_sizes=layer_sizes,
#         bottleneck_size=bottleneck_size,
#         dropout=dropout,
#         split_head=True,
#     )
#     # stack the logits and sigma for aleatoric loss
#     aleatoric = Stack(name='aleatoric')([logits, sigma])
#     # pass the logits through the Softmax activation to get probabilities
#     softmax = Activation('softmax', name='softmax')(logits)
#     # build the Tiramisu model
#     model = Model(inputs=[inputs], outputs=[softmax, sigma, aleatoric])

#     # if the weights file is not none, load the weights and transfer
#     if weights_file is not None:
#         from .tiramisu import tiramisu
#         # load the Tiramisu model to load from
#         transfer = tiramisu(image_shape, num_classes,
#             initial_filters=initial_filters,
#             growth_rate=growth_rate,
#             layer_sizes=layer_sizes,
#             bottleneck_size=bottleneck_size,
#             dropout=dropout,
#             learning_rate=learning_rate
#         )
#         transfer.load_weights(weights_file)
#         # the layers that have weights
#         is_weighted = lambda x: x.__class__.__name__ in {'Conv2D', 'BatchNormalization'}
#         # get the layers in the model
#         layers = [l for l in model.layers if is_weighted(l)]
#         # get the layers in the transfer mode
#         transfer_layers = [l for l in transfer.layers if is_weighted(l)]
#         # iterate over the layers in the original tiramisu network to transfer
#         for idx, layer in enumerate(transfer_layers):
#             layers[idx].set_weights(layer.get_weights())
#             layers[idx].trainable = not freeze_weights

#     # compile the model
#     model.compile(
#         optimizer=RMSprop(lr=learning_rate),
#         loss={
#             'softmax': build_categorical_crossentropy(class_weights),
#             'aleatoric': build_categorical_aleatoric_loss(samples)
#         },
#         metrics={'softmax': [build_categorical_accuracy(weights=class_weights)]},
#     )

#     return model


# explicitly define the outward facing API of this module
__all__ = [aleatoric_tiramisu.__name__]

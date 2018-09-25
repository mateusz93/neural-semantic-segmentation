"""An implementation of Tiramisu auto-encoder for semantic segmentation."""
from keras import backend as K
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import Concatenate
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import Dropout
from keras.layers import Input
from keras.layers import Lambda
from keras.layers import MaxPooling2D
from keras.models import Model
from keras.optimizers import RMSprop
from keras.regularizers import l2
from .layers import ContrastNormalization
from .metrics import metrics_for_segmentation
from .losses import build_weighted_categorical_crossentropy


# static arguments used for all convolution layers in Tiramisu models
_CONV = dict(
    kernel_initializer='he_uniform',
    kernel_regularizer=l2(1e-4),
)


def _dense_block(inputs,
    num_layers: int,
    num_filters: int,
    skip=None,
    dropout: float=0.2
):
    """
    Create a dense block for a given input tensor.

    Args:
        inputs: the input tensor to append this dense block to
        num_layers: the number of layers in this dense block
        num_filters: the number of filters in the convolutional layer
        skip: the skip mode of the dense block as a {str, None, Tensor}
            - 'downsample': the dense block is part of the down-sample side
            - None: the dense block is the bottleneck block bottleneck
            - a skip tensor: the dense block is part of the up-sample side
        dropout: the dropout rate to use per layer (None to disable dropout)

    Returns:
        a tensor with a new dense block appended to it

    """
    # create a placeholder list to store references to output tensors
    outputs = [None] * num_layers
    # if skip is a tensor, concatenate with inputs (upstream mode)
    if K.is_tensor(skip):
        # concatenate the skip with the inputs
        inputs = Concatenate()([inputs, skip])
    # copy a reference to the block inputs for later
    block_inputs = inputs
    # iterate over the number of layers in the block
    for idx in range(num_layers):
        # training=True to compute current batch statistics during inference
        # i.e., during training, validation, and testing
        x = BatchNormalization()(inputs, training=True)
        x = Activation('relu')(x)
        x = Conv2D(num_filters, kernel_size=(3, 3), padding='same', **_CONV)(x)
        if dropout is not None:
            x = Dropout(dropout)(x)
        # store the output tensor of this block to concatenate at the end
        outputs[idx] = x
        # concatenate the input with the outputs (unless last layer in block)
        if idx + 1 < num_layers:
            inputs = Concatenate()([inputs, x])

    # concatenate outputs to produce num_layers * num_filters feature maps
    x = Concatenate()(outputs)
    # if skip is 'downstream' concatenate inputs with outputs (downstream mode)
    if skip == 'downstream':
        x = Concatenate()([block_inputs, x])

    return x


def _transition_down_layer(inputs, dropout: float=0.2):
    """
    Create a transition layer for a given input tensor.

    Args:
        inputs: the input tensor to append this transition down layer to
        dropout: the dropout rate to use per layer (None to disable dropout)

    Returns:
        a tensor with a new transition down layer appended to it

    """
    # get the number of filters from the input activation maps
    num_filters = K.int_shape(inputs)[-1]
    # training=True to compute current batch statistics during inference
    # i.e., during training, validation, and testing
    x = BatchNormalization()(inputs, training=True)
    x = Activation('relu')(x)
    x = Conv2D(num_filters, kernel_size=(1, 1), padding='same', **_CONV)(x)
    if dropout is not None:
        x = Dropout(dropout)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    return x


def _transition_up_layer(inputs):
    """
    Create a transition up layer for a given input tensor.

    Args:
        inputs: the input tensor to append this transition up layer to

    Returns:
        a tensor with a new transition up layer appended to it

    """
    # get the number of filters from the number of activation maps
    return Conv2DTranspose(K.int_shape(inputs)[-1],
        kernel_size=(3, 3),
        strides=(2, 2),
        padding='same',
        **_CONV
    )(inputs)


def build_tiramisu(
    image_shape: tuple,
    num_classes: int,
    label_names: dict=None,
    class_weights=None,
    initial_filters: int=48,
    growth_rate: int=16,
    layer_sizes: list=[4, 5, 7, 10, 12],
    bottleneck_size: int=15,
    learning_rate: float=1e-3,
    contrast_norm: str='lcn'
) -> Model:
    """
    Build a DenseNet model for the given image shape.

    Args:
        image_shape: the image shape to create the model for
        num_classes: the number of classes to segment for (e.g. c)
        label_names: a dictionary mapping discrete labels to names for IoU
        class_weights: the weights for each class
        initial_filters: the number of filters in the first convolution layer
        growth_rate: the growth rate to use for the network (e.g. k)
        layer_sizes: a list with the size of each dense down-sample block.
                     reversed to determine the size of the up-sample blocks
        bottleneck_size: the number of convolutional layers in the bottleneck
        learning_rate: the learning rate for the RMSprop optimizer
        contrast_norm: the method of contrast normalization for inputs

    Returns:
        a compiled Tiramisu model

    """
    # the input block of the network
    inputs = Input(image_shape, name='Tiramisu_input')
    # assume 8-bit inputs and convert to floats in [0,1]
    x = Lambda(lambda x: x / 255.0, name='pixel_norm')(inputs)
    # apply contrast normalization if set
    if contrast_norm is not None:
        x = ContrastNormalization(method=contrast_norm, name=contrast_norm)(x)
    # the initial convolution layer
    x = Conv2D(initial_filters, kernel_size=(3, 3), padding='same', **_CONV)(x)
    # the down-sampling side of the network (keep outputs for skips)
    skips = [None] * len(layer_sizes)
    # iterate over the size for each down-sampling block
    for idx, size in enumerate(layer_sizes):
        skips[idx] = _dense_block(x, size, growth_rate, skip='downstream')
        x = _transition_down_layer(skips[idx])
    # the bottleneck of the network
    x = _dense_block(x, bottleneck_size, growth_rate)
    # the up-sampling side of the network (using kept outputs for skips)
    for idx, size in reversed(list(enumerate(layer_sizes))):
        x = _transition_up_layer(x)
        x = _dense_block(x, size, growth_rate, skip=skips[idx])
    # the classification block
    x = Conv2D(num_classes, kernel_size=(1, 1), padding='valid', **_CONV)(x)
    x = Activation('softmax')(x)
    # compile the graph
    model = Model(inputs=[inputs], outputs=[x])
    model.compile(
        optimizer=RMSprop(lr=learning_rate),
        loss=build_weighted_categorical_crossentropy(class_weights),
        metrics=metrics_for_segmentation(num_classes, label_names),
    )

    return model


# explicitly define the outward facing API of this module
__all__ = [build_tiramisu.__name__]

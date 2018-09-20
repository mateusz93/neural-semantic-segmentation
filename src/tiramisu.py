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
from .metrics import mean_iou
from .metrics import build_iou_for
from .losses import build_weighted_categorical_crossentropy


def _dense_block(inputs,
    num_layers: int,
    num_filters: int,
    mode=None,
    dropout: float=0.2
):
    """
    Create a dense block for a given input tensor.

    Args:
        inputs: the input tensor to append this dense block to
        num_layers: the number of layers in this dense block
        num_filters: the number of filters in the convolutional layer
        mode: the mode of the dense block as a {str, None, Tensor}
            - 'downsample': the dense block is part of the down-sample side
            - None: the dense block is the bottleneck block bottleneck
            - a skip tensor: the dense block is part of the up-sample side
        dropout: the dropout rate to use per layer (None to disable dropout)

    Returns:
        a tensor with a new dense block appended to it

    """
    # create a placeholder list to store references to output tensors
    outputs = [None] * num_layers
    # if mode is a tensor, we're in downstream mode
    if K.is_tensor(mode):
        # concatenate the mode (from skip connection) with the inputs
        inputs = Concatenate()([inputs, mode])
    # copy a reference to the block inputs for later
    block_inputs = inputs
    # iterate over the number of layers in the block
    for idx in range(num_layers):
        x = BatchNormalization()(inputs)
        x = Activation('relu')(x)
        x = Conv2D(num_filters,
            kernel_size=(3, 3),
            padding='same',
            kernel_initializer='he_uniform',
            kernel_regularizer=l2(1e-4),
        )(x)
        if dropout is not None:
            x = Dropout(dropout)(x)
        # store the output tensor of this block to concatenate at the end
        outputs[idx] = x
        # concatenate the input with the outputs (unless last layer in block)
        if idx + 1 < num_layers:
            inputs = Concatenate()([inputs, x])

    # concatenate outputs to produce num_layers * num_filters feature maps
    x = Concatenate()(outputs)
    # if we're in downstream mode
    if mode == 'downstream':
        # concatenate the block inputs with the outputs
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
    x = BatchNormalization()(inputs)
    x = Activation('relu')(x)
    x = Conv2D(K.int_shape(inputs)[-1],
        kernel_size=(1, 1),
        padding='same',
        kernel_initializer='he_uniform',
        kernel_regularizer=l2(1e-4),
    )(x)
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
    return Conv2DTranspose(K.int_shape(inputs)[-1],
        kernel_size=(3, 3),
        strides=(2, 2),
        padding='same',
        kernel_initializer='he_uniform',
        kernel_regularizer=l2(1e-4),
    )(inputs)


def build_tiramisu(
    image_shape: tuple,
    num_classes: int,
    label_names: dict=None,
    growth_rate: int=16,
    layer_sizes: list=[4, 5, 7, 10, 12],
    bottleneck_size: int=15,
    learning_rate: float=1e-3,
    class_weights=None,
    contrast_norm: str='lcn'
) -> Model:
    """
    Build a DenseNet model for the given image shape.

    Args:
        image_shape: the image shape to create the model for
        num_classes: the number of classes to segment for (e.g. c)
        label_names: a dictionary mapping discrete labels to names for IoU
        growth_rate: the growth rate to use for the network (e.g. k)
        layer_sizes: a list with the size of each dense down-sample block.
                     reversed to determine the size of the up-sample blocks
        bottleneck_size: the number of convolutional layers in the bottleneck
        learning_rate: the learning rate for the RMSprop optimizer
        class_weights: the weights for each class
        contrast_norm: the method of contrast normalization for inputs

    Returns:
        a Keras model of the 103 layer Tiramisu version of DenseNet

    """
    # the input block of the network
    inputs = Input(image_shape)
    # assume 8-bit inputs and convert to floats in [0,1]
    x = Lambda(lambda x: x / 255.0)(inputs)
    # apply contrast normalization if set
    if contrast_norm is not None:
        x = ContrastNormalization(method=contrast_norm)(x)
    # start the Tiramisu network
    x = Conv2D(48,
        kernel_size=(3, 3),
        padding='same',
        kernel_initializer='he_uniform',
        kernel_regularizer=l2(1e-4),
    )(x)
    # the down-sampling side of the network
    skips = [None] * len(layer_sizes)
    # iterate over the size for each down-sampling block
    for idx, size in enumerate(layer_sizes):
        skips[idx] = _dense_block(x, size, growth_rate, mode='downstream')
        x = _transition_down_layer(skips[idx])
    # the bottleneck of the network
    x = _dense_block(x, bottleneck_size, growth_rate)
    # the up-sampling side of the network
    for idx, size in reversed(list(enumerate(layer_sizes))):
        x = _transition_up_layer(x)
        x = _dense_block(x, size, growth_rate, mode=skips[idx])
    # the classification block
    x = Conv2D(num_classes,
        kernel_size=(1, 1),
        padding='valid',
        kernel_initializer='he_uniform',
        kernel_regularizer=l2(1e-4),
    )(x)
    x = Activation('softmax')(x)
    # compile the graph
    model = Model(inputs=[inputs], outputs=[x])
    model.compile(
        optimizer=RMSprop(lr=learning_rate),
        loss=build_weighted_categorical_crossentropy(class_weights),
        metrics=[
            'accuracy',
            mean_iou,
            *build_iou_for(list(range(num_classes)), label_names),
        ],
    )

    return model


# explicitly define the outward facing API of this module
__all__ = [build_tiramisu.__name__]

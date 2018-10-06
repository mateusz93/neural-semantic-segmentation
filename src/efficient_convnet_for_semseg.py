"""An implementation of "Efficient ConvNet for Semantic Segmentation"."""
from keras import backend as K
from keras.layers import Activation
from keras.layers import Add
from keras.layers import BatchNormalization
from keras.layers import Concatenate
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import Dropout
from keras.layers import Input
from keras.layers import Lambda
from keras.layers import MaxPooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from .losses import build_categorical_crossentropy
from .metrics import build_categorical_accuracy


# static arguments used for all convolution layers in SegNet
_CONV = dict(
    # TODO: there paper does not discuss initialization. what is best?
    # kernel_initializer='he_uniform',
    kernel_regularizer=l2(2e-4),
)


def _non_bottleneck_1D(x,
    kernel_sizes: list=[(3, 1), (1, 3), (3, 1), (1, 3)],
    batch_norms: list=[False, True, False, True],
    apply_dilations: list=[False, False, True, True],
    dilation_rate: int=1,
    dropout_rate: float=0.3
):
    """
    Create a new non-bottleneck 1D block for a given input tensor.

    Args:
        x: the input tensor to append this dense block to
        kernel_sizes: a list of tuples representing the kernel size for each
                      layer in the block
        batch_norms: a boolean vector of whether to use BN after the conv 2D
                     with kernel size of matching index
        apply_dilations: a boolean vector of whether to use given dilation
                         rate for convolutional layer with kernel size of
                         matching index
        dilation_rate: the dilation rate to apply to dilated conv layers
        dropout_rate: the rate to use for dropout at the end of the block
                      (before the add operation)

    Returns:
        a tensor with a new non-bottleneck 1D block appended to it

    """
    # initialize a reference to the input tensor
    inputs = x
    # get the number of filters from the shape of the input activations
    num_filters = K.int_shape(inputs)[-1]
    # iterate over the kernel sizes provided for each layer
    iterator = enumerate(zip(batch_norms, kernel_sizes, apply_dilations))
    for idx, (batch_norm, kernel_size, apply_dilation) in iterator:
        # pass the last outputs through a convolutional layer with kernel size
        x = Conv2D(num_filters,
            kernel_size=kernel_size,
            padding='same',
            dilation_rate=dilation_rate if apply_dilation else 1,
            **_CONV
        )(x)
        # if batch normalization is active for this layer, apply it
        if batch_norm:
            x = BatchNormalization()(x)
        # pass the outputs from convolution through ReLU nonlinearity if the
        # output _is not_ the last of the block
        if idx < len(kernel_sizes) - 1:
            x = Activation('relu')(x)
    # pass the values through dropout
    x = Dropout(dropout_rate)(x)
    # concatenate outputs of the last layer with the inputs to the block
    x = Add()([inputs, x])
    # pass the outputs from convolution through ReLU nonlinearity
    x = Activation('relu')(x)

    return x


def _encode_block(x, num_filters: int):
    """
    Create a new encoder block for a given input tensor.

    Args:
        x: the input tensor to append this dense block to
        num_filters: the number of filters to use in the convolutional layer

    Returns:
        a tensor with a new encoder block appended to it

    """
    # compute the actual number of filters based on the desired output
    # filters and actual input activation maps
    num_filters = num_filters - K.int_shape(x)[-1]
    # pass the x through a convolution
    x1 = Conv2D(num_filters,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding='same',
        **_CONV
    )(x)
    # split the inputs through a max pooling operation
    x2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    # concatenate the outputs from convolution and max pooling
    x = Concatenate()([x1, x2])
    # pass the outputs from convolution through ReLU nonlinearity
    x = Activation('relu')(x)

    return x


def _decode_block(x, num_filters: int):
    """
    Create a decoder block for a given input tensor.

    Args:
        x: the input tensor to append this decoder block to
        num_filters: the number of filters to use in the convolutional layer

    Returns:
        a tensor with a new decoder block appended to it

    """
    return Conv2DTranspose(num_filters,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding='same',
        **_CONV,
    )(x)


def _classify(x, num_classes: int):
    """
    Add a Softmax classification block to an input CNN.

    Args:
        x: the input tensor to append this classification block to (CNN)
        num_classes: the number of classes to predict with Softmax

    Returns:
        a tensor with dense convolution followed by Softmax activation

    """
    # dense convolution (1 x 1) to filter logits for each class
    x = Conv2DTranspose(num_classes,
        kernel_size=(1, 1),
        strides=(2, 2),
        **_CONV,
    )(x)
    # Softmax activation to convert the logits to probability vectors
    x = Activation('softmax')(x)

    return x


def efficient_convnet_for_semseg(image_shape: tuple, num_classes: int,
    class_weights=None,
    optimizer=Adam(lr=5e-4),
) -> Model:
    """
    Build an "Efficient ConvNet for Semantic Segmentation" model.

    Args:
        image_shape: the image shape to create the model for
        num_classes: the number of classes to segment for (e.g. c)
        class_weights: the weights for each class
        optimizer: the optimizer for training the network

    Returns:
        a compiled "Efficient ConvNet for Semantic Segmentation" model

    """
    # ensure the image shape is legal for the architecture
    div = int(2**3)
    for dim in image_shape[:-1]:
        # raise error if the dimension doesn't evenly divide
        if dim % div:
            msg = 'dimension ({}) must be divisible by {}'.format(dim, div)
            raise ValueError(msg)
    # the input block of the network
    inputs = Input(image_shape)
    # assume 8-bit inputs and convert to floats in [0,1]
    x = Lambda(lambda x: x / 255.0, name='pixel_norm')(inputs)
    # encoder
    x = _encode_block(x, 16)
    x = _encode_block(x, 64)
    for _ in range(5):
        x = _non_bottleneck_1D(x)
    x = _encode_block(x, 128)
    for dilation_rate in [2, 4, 8, 16] * 2:
        x = _non_bottleneck_1D(x, dilation_rate=dilation_rate)
    # decoder
    for filter_size in [64, 16]:
        x = _decode_block(x, filter_size)
        for _ in range(2):
            x = _non_bottleneck_1D(x)
    # classifier
    x = _classify(x, num_classes)
    # compile the model
    model = Model(inputs=[inputs], outputs=[x], name='SegNet')
    model.compile(
        optimizer=optimizer,
        loss=build_categorical_crossentropy(class_weights),
        metrics=[build_categorical_accuracy(weights=class_weights)],
    )

    return model


# explicitly define the outward facing API of this module
__all__ = [efficient_convnet_for_semseg.__name__]

"""An implementation of ENet."""
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
from keras.layers import PReLU
from keras.layers import SpatialDropout2D
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from .losses import build_categorical_crossentropy
from .metrics import build_categorical_accuracy


# static arguments used for all convolution layers in SegNet
_CONV = dict(
    # kernel_initializer='he_uniform',
    # kernel_regularizer=l2(2e-4),
)


def _initial_block(x, num_filters: int):
    """
    Create a new encoder block for a given input tensor.

    Args:
        x: the input tensor to append this dense block to
        num_filters: the number of filters to use in the convolutional layer

    Returns:
        a tensor with a new encoder block appended to it

    """
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
    # apply batch normalization and PReLU activation
    x = BatchNormalization()(x)
    x = PReLU()(x)

    return x


def _bottleneck_module(x, num_filters: int, conv,
    kernel_size=(1, 1),
    strides=(1, 1),
    dropout_rate: float=0.01,
    **conv_kwargs,
):
    """
    Create a new bottleneck module.

    Args:
        x: the input tensor to append this bottleneck module to
        num_filters:
        conv:
        strides:
        dropout_rate:
        conv_kwargs:

    Returns:
        a tensor with a new bottleneck module appended to it

    """
    # 1 x 1 projection
    x = Conv2D(num_filters,
        kernel_size=kernel_size,
        strides=strides,
        padding='same',
        use_bias=False,
        **_CONV
    )(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    # the bottleneck layer
    x = conv(x, num_filters, **conv_kwargs)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    # 1 x 1 expansion
    x = Conv2D(num_filters,
        kernel_size=(1, 1),
        padding='same',
        use_bias=False,
        **_CONV
    )(x)
    # regularization
    x = BatchNormalization()(x)
    x = SpatialDropout2D(dropout_rate)(x)

    return x


def _downsample_module(x, num_filters: int, dropout_rate: float=0.01):
    """
    .

    Args:
        x:
        num_filters:
        dropout_rate:

    Returns:


    """
    # standard stream through the bottleneck module
    x1 = _bottleneck_module(x, num_filters, _conv,
        kernel_size=(2, 2),
        strides=(2, 2),
        dropout_rate=dropout_rate,
    )
    # max pooling and appropriate padding
    x2 = MaxPooling2D()(x)
    # zero padding is surprisingly difficult to implement and bad. a learned
    # linear transformation should work better anyway
    x2 = Conv2D(num_filters,
        kernel_size=(1, 1),
        padding='same',
        use_bias=False,
        **_CONV
    )(x2)
    # adding and activation
    x = Add()([x1, x2])
    x = PReLU()(x)

    return x


def _conv(x, num_filters: int, dilation_rate: int=1):
    """
    .

    Args:
        x:
        num_filters:
        dilation_rate:

    Returns:


    """
    x = Conv2D(num_filters,
        kernel_size=(3, 3),
        padding='same',
        dilation_rate=dilation_rate,
        **_CONV
    )(x)

    return x


def _asymmetric_conv(x, num_filters: int, asymmetric_kernel: int=5):
    """
    .

    Args:
        x:
        num_filters:
        asymmetric_kernel:

    Returns:


    """
    # make an iterator for the kernel sizes for asymmetric convolution
    kernel_sizes = enumerate([(asymmetric_kernel, 1), (1, asymmetric_kernel)])
    # apply the kernel over both dimensions (asymmetric)
    for idx, kernel_size in kernel_sizes:
        x = Conv2D(num_filters,
            kernel_size=kernel_size,
            padding='same',
            use_bias=bool(idx),
            **_CONV,
        )(x)

    return x


def _deconv(x, num_filters: int):
    """
    .

    Args:
        x:
        num_filters:

    Returns:


    """
    # perform deconvolution with stride of 2 x 2
    return Conv2DTranspose(num_filters,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding='same',
        **_CONV
    )(x)


def _encode(x):
    """
    .

    Args:
        x:

    Returns:


    """
    # 1.0: down-sampling
    x = _downsample_module(x, 64, dropout_rate=0.01)
    # 4x 1.x
    for _ in range(4):
        x = _bottleneck_module(x, 64, _conv, dropout_rate=0.01)
    # 2.0: down-sampling
    x = _downsample_module(x, 128, dropout_rate=0.1)
    # (repeat section 2 without 2.0)
    for _ in range(2):
        # 2.1
        x = _bottleneck_module(x, 128, _conv, dropout_rate=0.1)
        # 2.2: dilated 2
        x = _bottleneck_module(x, 128, _conv, dropout_rate=0.1, dilation_rate=2)
        # 2.3: asymmetric 5
        x = _bottleneck_module(x, 128, _asymmetric_conv, dropout_rate=0.1)
        # 2.4: dilated 4
        x = _bottleneck_module(x, 128, _conv, dropout_rate=0.1, dilation_rate=4)
        # 2.5
        x = _bottleneck_module(x, 128, _conv, dropout_rate=0.1)
        # 2.6: dilated 8
        x = _bottleneck_module(x, 128, _conv, dropout_rate=0.1, dilation_rate=8)
        # 2.7: asymmetric 5
        x = _bottleneck_module(x, 128, _asymmetric_conv, dropout_rate=0.1)
        # 2.8: dilated 16
        x = _bottleneck_module(x, 128, _conv, dropout_rate=0.1, dilation_rate=16)

    return x


def _decode(x):
    """
    .

    Args:
        x:

    Returns:


    """
    # 4.0: upsampling
    x = _bottleneck_module(x, 64, _deconv, dropout_rate=0.1)
    # 4.1
    x = _bottleneck_module(x, 64, _conv, dropout_rate=0.1)
    # 4.2
    x = _bottleneck_module(x, 64, _conv, dropout_rate=0.1)
    # 5.0: upsampling
    x = _bottleneck_module(x, 16, _deconv, dropout_rate=0.1)
    # 5.1
    x = _bottleneck_module(x, 16, _conv, dropout_rate=0.1)

    return x


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
        kernel_size=(2, 2),
        strides=(2, 2),
        **_CONV,
    )(x)
    # Softmax activation to convert the logits to probability vectors
    x = Activation('softmax')(x)

    return x


def enet(image_shape: tuple, num_classes: int,
    class_weights=None,
    optimizer=Adam(lr=5e-4),
) -> Model:
    """
    Build an ENet model.

    Args:
        image_shape: the image shape to create the model for
        num_classes: the number of classes to segment for (e.g. c)
        class_weights: the weights for each class
        optimizer: the optimizer for training the network

    Returns:
        a compiled ENet model

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
    # initial block (16 output filters)
    x = _initial_block(x, 16)
    x = _encode(x)
    x = _decode(x)
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
__all__ = [enet.__name__]

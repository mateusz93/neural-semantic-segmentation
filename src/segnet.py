"""An implementation of SegNet auto-encoder for semantic segmentation."""
from keras.models import Model
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import Activation
from keras.layers import Lambda
from keras.optimizers import SGD
from keras.regularizers import l2
from .layers import MemorizedMaxPooling2D
from .layers import MemorizedUpsampling2D
from .iou import mean_iou
from .iou import build_iou_for


def conv_bn_relu(x, num_filters: int):
    """
    Append a conv + batch normalization + relu block to an input tensor.

    Args:
        x: the input tensor to append this dense block to
        num_filters: the number of filters in the convolutional layer

    Returns:
        an updated graph with conv + batch normalization + relu block added

    """
    x = Conv2D(num_filters,
        kernel_size=(3, 3),
        padding='same',
        kernel_initializer='he_uniform',
        kernel_regularizer=l2(1e-4),
    )(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


def downsample(x, num_conv: int, num_filters: int):
    """
    Append a down-sampling block with a given size and number of filters.

    Args:
        x: the input tensor to append this down-sample block to
        num_conv: the number of convolutional blocks to use before pooling
        num_filters: the number of filters in each convolutional layer

    Returns:
        a tuple of:
        - an updated graph with num_conv conv blocks followed by max pooling
        - the pooling layer to get indexes from for up-sampling

    """
    for _ in range(num_conv):
        x = conv_bn_relu(x, num_filters)
    pool = MemorizedMaxPooling2D(pool_size=(2, 2), strides=(2, 2))
    x = pool(x)
    return x, pool


def upsample(x, pool: MemorizedMaxPooling2D, num_conv: int, num_filters: int):
    """
    Append an up-sampling block with a given size and number of filters.

    Args:
        x: the input tensor to append this up-sample block to
        pool: the corresponding memorized pooling layer to reference indexes
        num_conv: the number of convolutional blocks to use before pooling
        num_filters: the number of filters in each convolutional layer

    Returns:
        an updated graph with up-sampling followed by num_conv conv blocks

    """
    x = MemorizedUpsampling2D(pool=pool)(x)
    for _ in range(num_conv):
        x = conv_bn_relu(x, num_filters)
    return x


def classification(x, num_classes: int):
    """
    Add a Softmax classification block to an input CNN.

    Args:
        x: the input tensor to append this classification block to (CNN)
        num_classes: the number of classes to predict with Softmax

    Returns:
        an updated graph with dense convolution followed by Softmax activation

    """
    x = Conv2D(num_classes,
        kernel_size=(1, 1),
        padding='same',
        kernel_initializer='he_uniform',
        kernel_regularizer=l2(1e-4),
    )(x)
    x = BatchNormalization()(x)
    x = Activation('softmax')(x)
    return x


def build_segnet(
    image_shape: tuple,
    num_classes: int,
    label_names: dict=None,
    optimizer=SGD(lr=0.1, momentum=0.9)
) -> Model:
    """
    Build a SegNet model for the given image shape.

    Args:
        image_shape: the image shape to create the model for
        num_classes: the number of classes to segment for (e.g. c)
        label_names: a dictionary mapping discrete labels to names for IoU
        optimizer: the optimizer for training the network

    Returns:
        a Keras model of the 103 layer Tiramisu version of DenseNet

    """
    # the input block of the network
    inputs = Input(image_shape)
    # assume 8-bit inputs and convert to floats in [0,1]
    x = Lambda(lambda x: x / 255.0)(inputs)
    # down-sample
    x, p1 = downsample(x, num_conv=2, num_filters=64)
    x, p2 = downsample(x, num_conv=2, num_filters=128)
    x, p3 = downsample(x, num_conv=3, num_filters=256)
    x, p4 = downsample(x, num_conv=3, num_filters=512)
    x, p5 = downsample(x, num_conv=3, num_filters=512)
    # up-sample
    x = upsample(x, p5, num_conv=3, num_filters=512)
    x = upsample(x, p4, num_conv=3, num_filters=256)
    x = upsample(x, p3, num_conv=3, num_filters=128)
    x = upsample(x, p2, num_conv=2, num_filters=64)
    x = upsample(x, p1, num_conv=2, num_filters=64)
    # classification
    x = classification(x, num_classes)
    # compile the graph
    model = Model(inputs=[inputs], outputs=[x])
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            mean_iou,
            *build_iou_for(list(range(num_classes)), label_names),
        ],
    )

    return model


# explicitly define the outward facing API of this module
__all__ = [build_segnet.__name__]

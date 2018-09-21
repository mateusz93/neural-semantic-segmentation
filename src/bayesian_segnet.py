"""An implementation of SegNet auto-encoder for semantic segmentation."""
from keras.applications.vgg16 import VGG16
from keras.layers import Conv2D
from keras.layers import Input
from keras.layers import Lambda
from keras.models import Model
from keras.optimizers import SGD
from .layers import ContrastNormalization
from .layers import Mean
from .layers import MonteCarlo
from .layers import MonteCarloDropout
from .layers import Var
from .losses import build_weighted_categorical_crossentropy
from .metrics import mean_iou
from .metrics import build_iou_for
from .segnet import classify
from .segnet import decode
from .segnet import encode


def build_bayesian_segnet(
    image_shape: tuple,
    num_classes: int,
    label_names: dict=None,
    optimizer=SGD(lr=0.001, momentum=0.9),
    dropout_rate: float=0.5,
    pretrain_encoder: bool=True,
    class_weights=None,
    contrast_norm: str='lcn'
) -> Model:
    """
    Build a SegNet model for the given image shape.

    Args:
        image_shape: the image shape to create the model for
        num_classes: the number of classes to segment for (e.g. c)
        label_names: a dictionary mapping discrete labels to names for IoU
        optimizer: the optimizer for training the network
        dropout_rate: the dropout rate to use at test time
        pretrain_encoder: whether to initialize the encoder from VGG16
        class_weights: the weights for each class
        contrast_norm: the method of contrast normalization for inputs

    Returns:
        a compiled Bayesian SegNet model

    """
    # the input block of the network
    inputs = Input(image_shape)
    # assume 8-bit inputs and convert to floats in [0,1]
    x = Lambda(lambda x: x / 255.0, input_shape=image_shape)(inputs)
    # apply contrast normalization if set
    if contrast_norm is not None:
        x = ContrastNormalization(method=contrast_norm)(x)
    # encoder
    x, p1 = encode(x, 2 * [64])
    x, p2 = encode(x, 2 * [128])
    x, p3 = encode(x, 3 * [256])
    x = MonteCarloDropout(dropout_rate)(x)
    x, p4 = encode(x, 3 * [512])
    x = MonteCarloDropout(dropout_rate)(x)
    x, p5 = encode(x, 3 * [512])
    x = MonteCarloDropout(dropout_rate)(x)
    # decoder
    x = decode(x, p5, 3 * [512])
    x = MonteCarloDropout(dropout_rate)(x)
    x = decode(x, p4, [512, 512, 256])
    x = MonteCarloDropout(dropout_rate)(x)
    x = decode(x, p3, [256, 256, 128])
    x = MonteCarloDropout(dropout_rate)(x)
    x = decode(x, p2, [128, 64])
    x = decode(x, p1, [64])
    # classifier
    x = classify(x, num_classes)
    # compile the graph
    model = Model(inputs=[inputs], outputs=[x], name='SegNet')
    model.compile(
        optimizer=optimizer,
        loss=build_weighted_categorical_crossentropy(class_weights),
        metrics=[
            'accuracy',
            mean_iou,
            *build_iou_for(list(range(num_classes)), label_names),
        ],
    )
    # if transfer learning from ImageNet is disabled, return the model as is
    if not pretrain_encoder:
        return model
    # load the pre-trained VGG16 model using ImageNet weights
    vgg16 = VGG16(weights='imagenet', include_top=False)
    # extract all the convolutional layers (encoder layers) from VGG16
    vgg16_conv = [layer for layer in vgg16.layers if isinstance(layer, Conv2D)]
    # extract all convolutional layers from SegNet, the first len(vgg16_conv)
    # layers in this list are architecturally congruent with the layers in
    # vgg16_conv by index
    model_conv = [layer for layer in model.layers if isinstance(layer, Conv2D)]
    # iterate over the VGG16 layers and replace the SegNet encoder weights
    for idx, layer in enumerate(vgg16_conv):
        model_conv[idx].set_weights(layer.get_weights())

    return model


def wrap_monte_carlo(model, num_samples: int=40):
    """
    Return a model to estimate the mean/var of another model with Monte Carlo.

    Args:
        model: the model to wrap with a Monte Carlo estimation
        num_samples: the number of samples of the model to estimate the mean

    Returns:
        a new model estimating the mean output of the given model

    """
    # the inputs for the Monte Carlo model (ignoring batch size)
    inputs = model.inputs
    # sample from the model for the given number of samples in Monte Carlo
    samples = MonteCarlo(model, num_samples)(inputs)
    # calculate the mean and variance of the Monte Carlo samples (axis -1)
    mean = Mean(name='mc')(samples)
    var = Mean(name='var')(Var()(samples))
    # build the epistemic uncertainty model
    mc_model = Model(inputs=inputs, outputs=[mean, var])
    # compile the model (optimizer is arbitrary, this is test only)
    mc_model.compile(optimizer='sgd', loss={'mc': model.loss}, metrics={'mc': model.metrics})

    return mc_model


# explicitly define the outward facing API of this module
__all__ = [build_bayesian_segnet.__name__, wrap_monte_carlo.__name__]

"""Custom metrics for Keras."""
from .categorical_accuracy import build_categorical_accuracy
from .iou import build_iou_for
from .iou import build_mean_iou
from .mean_per_class_accuracy import build_mean_per_class_accuracy


def metrics_for_segmentation(num_classes: int,
    label_names: dict=None,
    weights=None,
) -> list:
    """
    Return a list of metrics to use for semantic segmentation evaluation.

    Args:
        num_classes: the number of classes to segment for (e.g. c)
        label_names: a dictionary mapping discrete labels to names for IoU
        weights: optional weights for the metrics

    Returns:
        a list of metrics:
        - global accuracy
        - mean per class accuracy
        - mean IoU
        - num_classes IoU scores for each individual class label

    """
    categorical_accuracy = build_categorical_accuracy(weights=weights)
    mean_per_class_accuracy = build_mean_per_class_accuracy(weights=weights)
    mean_iou = build_mean_iou(weights=weights)
    # make an IoU metric for each class (that has nonzero weight)
    ious = []
    for label in range(num_classes):
        # if weights are given and defined as 0 for this label, continue
        if weights is not None and weights[label] <= 0:
            continue
        # build an I/U metric and add it to the list
        ious += [build_iou_for(label, label_names[label])]
    # return the cumulative list of metrics
    return [categorical_accuracy, mean_per_class_accuracy, mean_iou, *ious]


# explicitly define the outward facing API of this package
__all__ = [
    build_categorical_accuracy.__name__,
    build_iou_for.__name__,
    build_mean_iou.__name__,
    build_mean_per_class_accuracy.__name__,
    metrics_for_segmentation.__name__,
]

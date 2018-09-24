"""Custom metrics for Keras."""
from .iou import build_iou_for
from .iou import mean_iou
from .mean_per_class_accuracy import mean_per_class_accuracy


def metrics_for_segmentation(num_classes: int, label_names: dict=None):
    """
    Return a list of metrics to use for semantic segmentation evaluation.

    Args:
        num_classes: the number of classes to segment for (e.g. c)
        label_names: a dictionary mapping discrete labels to names for IoU

    Returns:
        a list of metrics:
        - global accuracy
        - mean per class accuracy
        - mean IoU
        - num_classes IoU scores for each individual class label

    """
    # make an IoU metric for each class
    ious = build_iou_for(list(range(num_classes)), label_names)
    # return the cumulative list of metrics
    return ['accuracy', mean_per_class_accuracy, mean_iou, *ious]


# explicitly define the outward facing API of this package
__all__ = [
    build_iou_for.__name__,
    mean_iou.__name__,
    mean_per_class_accuracy.__name__,
    metrics_for_segmentation.__name__,
]

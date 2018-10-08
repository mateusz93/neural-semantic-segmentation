"""A Method to evaluate segmentation models using NumPy metrics."""
import numpy as np
import pandas as pd
from tqdm import tqdm
from .metrics.evaluation_metrics import metrics


def evaluate(model, generator, steps: int,
    mask: np.ndarray=None,
    code_map: dict=None
) -> list:
    """
    Evaluate a segmentation model and return a DataFrame of metrics.

    Args:
        model: the model to generate predictions from
        generator: the generate to get data from
        steps: the number of steps to go through the generator
        mask: the mask to exclude certain labels
        code_map: a mapping of probability vector indexes to label names

    Returns:
        a DataFrame with the metrics from the generator data

    """
    y_true = [None] * steps
    y_pred = [None] * steps
    # iterate over the number of steps to generate data
    for step in tqdm(range(steps), unit='step'):
        # get the batch of data from the generator
        imgs, true = next(generator)
        # if true is a tuple or list, take the first target value
        y_true[step] = true[0] if isinstance(true, (tuple, list)) else true
        # get predictions from the network
        pred = model.predict(imgs)
        # if pred is a tuple or list, take the first network output
        y_pred[step] = pred[0] if isinstance(pred, (tuple, list)) else pred
    # convert the batch of targets to a NumPy tensor
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # calculate the metrics from the predicted and ground truth values
    _metrics = metrics(y_true, y_pred, mask)
    accuracy, mean_per_class_accuracy, mean_iou, iou = _metrics
    # build a dictionary to store metrics in
    _metrics = {
        '#1#Accuracy': accuracy,
        '#2#Mean Per Class Accuracy': mean_per_class_accuracy,
        '#3#Mean I/U': mean_iou,
    }
    # set the label map to an empty dictionary if it's None
    code_map = code_map if code_map is not None else dict()
    # iterate over the labels and I/Us in the vector of I/Us
    for label, iou_c in enumerate(iou):
        # if the value is in the mask, add it's value to the metrics dictionary
        if mask[label]:
            _metrics[code_map.get(label, str(label))] = iou_c
    # create a series with the metrics data
    _metrics = pd.Series(_metrics).sort_index()
    # replace the markers that force the core metrics to the top
    _metrics.index = _metrics.index.str.replace(r'#\d#', '')
    # convert the series to a DataFrame before returning
    return pd.DataFrame(_metrics, columns=['Value'])


# explicitly define the outward facing API of this module
__all__ = [evaluate.__name__]

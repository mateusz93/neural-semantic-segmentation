"""Methods to evaluate models using NumPy metrics."""
import numpy as np
import pandas as pd
from tqdm import tqdm
from .metrics.numpy_metrics import categorical_accuracy
from .metrics.numpy_metrics import iou
from .metrics.numpy_metrics import mean_iou
from .metrics.numpy_metrics import mean_per_class_accuracy


def evaluate(model, generator, steps: int,
    mask: np.ndarray=None,
    label_map: dict=None
) -> list:
    """
    Evaluate a segmentation model and return a DataFrame of metrics.

    Args:
        model: the model to generate predictions from
        generator: the generate to get data from
        steps: the number of steps to go through the generator
        mask: the mask to exclude certain labels
        label_map: a mapping of label names to probability vector indexes

    Returns:
        a DataFrame with the metrics from the generator data

    """
    y_true = [None] * steps
    y_pred = [None] * steps
    # iterate over the number of steps to generate data
    for step in tqdm(range(steps), unit='step'):
        # get the batch of data from the generator
        imgs, y_true[step] = next(generator)
        # get predictions from the network
        pred = model.predict(imgs)
        # if pred is a tuple, take the first network output
        if isinstance(pred, tuple):
            pred = pred[0]
        # store the prediction
        y_pred[step] = pred
    # convert the batch of targets to a NumPy tensor
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # create a dictionary with the metrics
    metrics = {
        '#1#Accuracy': categorical_accuracy(y_true, y_pred, mask=mask),
        '#2#Mean Per Class Accuracy': mean_per_class_accuracy(y_true, y_pred, mask=mask),
        '#3#Mean I/U': mean_iou(y_true, y_pred, mask=mask)
    }
    # calculate I/U for individual labels if the names are not none
    if label_map is not None:
        for name, label in label_map.items():
            # get I/U if the weights are non zero
            if mask[label]:
                metrics[name] = iou(y_true, y_pred, label)
    # create a series with the metrics data
    series = pd.Series(metrics).sort_index()
    # replace the markers that force the core metrics to the top
    series.index = series.index.str.replace(r'#\d#', '')
    # convert the series to a DataFrame before returning
    return pd.DataFrame(series, columns=['Test'])


# explicitly define the outward facing API of this module
__all__ = [evaluate.__name__]

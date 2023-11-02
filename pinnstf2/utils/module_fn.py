import tensorflow as tf

from typing import Dict, List, Union, Optional, Tuple


def sse(loss: tf.Tensor,
        preds: Dict[str, tf.Tensor],
        target: Union[Dict[str, tf.Tensor], None] = None,
        keys: Union[List[str], None] = None,
        mid: Union[int, None] = None) -> tf.Tensor:
    """Calculate the sum of squared errors (SSE) loss for given predictions and optional targets.

    :param loss: Loss variable.
    :param preds: Dictionary containing prediction tensors for different keys.
    :param target: Dictionary containing target tensors (optional).
    :param keys: List of keys for which to calculate SSE loss (optional).
    :param mid: Index to separate predictions for mid-point calculation (optional).
    :return: Calculated SSE loss.
    """
    
    if keys is None:
        return loss

    for key in keys:
        if target is None and mid is None:
            loss = loss + tf.reduce_sum(tf.square(preds[key]))
        elif target is None and mid is not None:
            loss = loss + tf.reduce_sum(tf.square(preds[key][:mid] - preds[key][mid:]))
        elif target is not None:
            loss = loss + tf.reduce_sum(tf.square(preds[key] - target[key]))

    return loss

def mse(loss: tf.Tensor,
        preds: Dict[str, tf.Tensor],
        target: Union[Dict[str, tf.Tensor], None] = None,
        keys: Union[List[str], None] = None,
        mid: Union[int, None] = None) -> tf.Tensor:
    """Calculate the mean squared error (MSE) loss for given predictions and optional targets.

    :param loss: Loss variable.
    :param preds: Dictionary containing prediction tensors for different keys.
    :param target: Dictionary containing target tensors (optional).
    :param keys: List of keys for which to calculate SSE loss (optional).
    :param mid: Index to separate predictions for mid-point calculation (optional).
    :return: Calculated MSE loss.
    """
    
    if keys is None:
        return loss

    for key in keys:
        if target is None:
            loss = loss + tf.reduce_mean(tf.square(preds[key]))
        elif target is None and mid is not None:
            loss = loss + tf.reduce_mean(tf.square(preds[key][:mid] - preds[key][mid:]))
        elif target is not None:
            loss = loss + tf.reduce_mean(tf.square(preds[key] - target[key]))

    return loss


def relative_l2_error(preds, target):
    """Calculate the relative L2 error between predictions and target tensors.

    :param preds: Predicted tensors.
    :param target: Target tensors.
    :return: Relative L2 error value.
    """
    
    #return tf.sqrt(tf.reduce_mean(tf.square(preds - target))/tf.reduce_mean(tf.square(target)))
    return tf.sqrt(tf.reduce_mean(tf.square(preds - target))/tf.reduce_mean(tf.square(target - tf.reduce_mean(target))))


def fix_extra_variables(trainable_variables, extra_variables, dtype):
    """Convert extra variables to tf tensors with gradient tracking. These variables are
    trainables in inverse problems.

    :param extra_variables: Dictionary of extra variables to be converted.
    :return: Dictionary of converted extra variables as tf tensors with gradients.
    """
    
    if extra_variables is None:
        return trainable_variables, None
    extra_variables_dict = {}
    for key in extra_variables:
        variable =  tf.Variable(extra_variables[key], dtype=tf.float32, trainable=True)
        extra_variables_dict[key] = variable
        trainable_variables.append(variable)
    return trainable_variables, extra_variables_dict

def fix_predictions(preds_dict):
    for sol_key, pred in preds_dict.items():
        preds_dict[sol_key] = pred.numpy()
    return preds_dict
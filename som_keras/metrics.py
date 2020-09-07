import numpy as np
import tensorflow as tf


def quantization_error(model, data, reduction='mean'):
    '''
    Returns reduced quantization error of the input data (that is
    the distance to the closest weight for each data point) using
    the provided model.
        model: SOM model to use for quantization.
        data: (tf Tensor) data for which to compute quantization
               error.
        reduction: (string) One of 'mean', 'sum'
    returns:
        red_q_error: reduced quantization error
    '''
    assert(reduction in ['mean', 'sum'])
    dists = model._distance_matrix(data)
    min_dists = tf.math.reduce_min(dists, axis=1)
    if reduction=='mean':
        red_q_error = tf.math.reduce_mean(min_dists, axis=0)
    else:
        red_q_error = tf.math.reduce_sum(min_dists, axis=0)
    return red_q_error


def q_error(y_true, y_pred, reduction='mean'):
    '''
    Returns reduced quantization error of the input data (that is
    the distance to the closest weight for each data point) using
    predictions
        y_true: (tf Tensor) input data
        y_pred: (tf Tensor) prediction (quantized data)
    returns:
        red_q_error: (float) reduced quantization error
    '''
    assert(reduction in ['mean', 'sum'])
    error = tf.math.reduce_euclidean_norm((y_true-y_pred), axis=1)
    # Reduce
    if reduction=='mean':
        red_q_error = tf.math.reduce_mean(error, axis=0)
    else:
        red_q_error = tf.math.reduce_sum(error, axis=0)
    return red_q_error


def topographic_error(model, data):
    '''
    Computes the fraction of data points for which the first
    and second BMUs are not adjacent.
        model: SOM model to evaluate
        data: (tf Tensor) data used to evaluate it
    returns:
        topographical error: proportion of non adjacent BMUs
    '''
    data = data.numpy()
    adj_mat = model._construct_adjmat(True)
    
    # Compute all distances
    dists = model._distance_matrix(data)
    # Get closest and second clostest BMUs
    sorted_rows = np.apply_along_axis(lambda x: np.argsort(x), 1, dists)
    bmus = sorted_rows[:, 0]
    bmus2 = sorted_rows[:, 1]
    
    # Count adjacent ones
    correct = np.sum(adj_mat[bmus, bmus2])
    return 1-correct/data.shape[0]

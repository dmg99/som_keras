import tensorflow as tf

def quantization_error(model, data, reduction='mean'):
    '''
    Returns mean quantization error of the input data (that is
    the distance to the closest weight for each data point)
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

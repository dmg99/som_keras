import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from scipy.spatial import distance_matrix
from sklearn.decomposition import PCA
from scipy.sparse import lil_matrix, csr_matrix

class SOM(keras.Model):
    '''
    Keras implementation of a Self-Organizing map.
    In order to extract the trained weights, us
    '''
    def __init__(self, dim_len, sigma=1, data=None, max_dim_len=10, **kwargs):
        '''
        Initializes SOM Model. In order to initialize the values
        using a dataset please use one of the functions
        initialize_random or initialize_with_pca. Otherwise,
        values will be initialized completely at random.
        
        dim_len: if it is an int, it uses a square lattice with
                 that side length. If it is a list with length 2,
                 it uses those values as the grid's sides. If None,
                 see data argument description.
        sigma: (float) sigma for the gaussian kernel used
               in the neighbourhood function.
        data: (tf Tensor) only used if dim_len is None. In that case,
              grid shape is determined using the two largest
              eigenvals from the data's PCA
        max_dim_len: (int) maximum possible value for each dimension
                     if shape is computed using PCA.
        '''
        super(SOM, self).__init__(**kwargs)
        # Depending on dim_len, use square or rectangular lattice
        if type(dim_len) == int:
            self.dim_len = [dim_len, dim_len]
        elif dim_len is None:
            assert(data is not None)
            self.dim_len = None
            self.initialize_with_pca(data, False, max_dim_len)
        else:
            assert(type(dim_len) == list)
            assert(len(dim_len) == 2)
            self.dim_len = dim_len

        self.sigma = tf.constant(sigma, dtype=tf.float64)
        
        # While initialize_with_pca is not called
        self.init_rand = True
         
        # Loss used in training
        self.mse_loss = keras.losses.MeanSquaredError()
        
        # Whether to return position on the grid (True) or weight
        self.get_pos = False

        # Labels for classification
        self.counts = None
        self.labels = None


    def build(self, input_shape):
        # If when an input is passed, the weights have
        # not been set, initialize as random.
        if self.init_rand:
            self.w = self.add_weight(
                shape=(input_shape[-1], self.dim_len[0]*self.dim_len[1]),
                initializer='random_normal',
                trainable=True,
                dtype=tf.float64,
                name='SOM-w'
            )


    def initialize_random(self, input, seed=None):
        '''
        Initializes SOM weights using random data samples.
        input: (tf.Tensor) data from which initialization is performed.
               Should be (n_samples x n_features).
        '''
        if seed is not None:
            tf.random.set_seed(seed)
        # Draw as many samples as weights in the grid
        num_w = int(self.dim_len[0]*self.dim_len[1])
        idxs = tf.range(input.shape[0])
        idxs = tf.random.shuffle(idxs)
        first_idxs = idxs[:num_w]
        weights = tf.gather(input, first_idxs, axis=0)
        
        # Initialize weights
        self.w = tf.Variable(tf.transpose(weights), trainable=True,
                             name='SOM-w')
        self.init_rand=False


    def initialize_with_pca(self, input, init_weights=True,
                            max_dim_len=10):
        '''
        Uses PCA to determine the shape of the SOM grid.
        Also, if init_weights is True, initializes SOM weights 
        by uniformly sampling the first two principal dimensions 
        of the input data.
        input: (tf Tensor) data from which initialization is performed.
               Should be n_samples x n_features.
        init_weights: (bool) whether to initialize weights or just compute
                      grid dimensions (if dim_len is None)
        max_dim_len: (int) maximum value allowed for the grid dimensions.
                     This can be used to scale the grid's size,
                     while using grid lengths proportional to the
                     first to eigenvalues of the PCA.
        '''
        # Get 2 dimensions with highest variance using PCA
        input_np = input.numpy()
        _, nfeatures = input_np.shape

        # Fit PCA
        pca = PCA(n_components=2)
        pca.fit(input)
        eigenvecs = pca.components_
        eigenvals = np.sqrt(pca.explained_variance_)
        if self.dim_len is None:
            self.dim_len = [max_dim_len, max_dim_len]
            props = eigenvals/np.sum(eigenvals)
            self.dim_len = np.ceil(self.dim_len*props).astype(int)
        
        if init_weights:
            # The limits will be the eigenvectors times the sd 
            # in each dimension 
            midpoint0 = self.dim_len[1]//2
            midpoint1 = self.dim_len[0]//2
            limits = np.zeros(eigenvecs.shape)
            
            limits[0] = eigenvecs[0]*eigenvals[0]**2/midpoint0
            limits[1] = eigenvecs[1]*eigenvals[1]**2/midpoint1

            # Uniformly sample initial weights 
            # along 2 first principal components 
            idxs = np.indices((self.dim_len[0], self.dim_len[1])).reshape(2, -1)
            # center indices at 0
            idxs[0] = idxs[0]-midpoint0
            idxs[1] = idxs[1]-midpoint1
            # The weight at position (i, j) is (i-size/2)*limit[0]+
            # + (j-size/2)*limit[1]
        
            init_w = tf.convert_to_tensor(np.transpose(limits) @ idxs,
                                          dtype=tf.float64)
            self.w = tf.Variable(init_w, trainable=True, name='SOM-w')
            self.init_rand = False

    def get_weights(self):
        '''
        Returns the SOM weights as numpy array of shape
        (n_neurons x n_features).
        '''
        return self.w.numpy().T


    def switch_preds(self):
        '''
        Switches prediction mode.
        If get_pos is True, it will predicted the position
        on the grid (it returns the flattened index)

        If get_pos is False it will return the most similar
        weight on the grid.
        '''
        self.get_pos = not self.get_pos


    def get_labels(self):
        '''
        If neurons on the SOM grid have been classified using
        classification.classify_SOM, returns class of each point on
        the grid.
        returns:
            labels: (np array) Array with length equal to grid size.
        '''
        assert(self.labels is not None)
        return self.labels


    def _gaussian_kernel(self, pos):
        '''
        Generates all gaussian kernels needed for the loss function
        '''
        # Get vector of spatial indices
        idxs = np.indices((self.dim_len[0], self.dim_len[1]))
        idxs = idxs.reshape(2, -1)
        idxs = tf.convert_to_tensor(idxs)

        # Each value of the kernel is computed as exp(dist^2/sigma)
        # where dist is the distance between position (i, j) and the center
        pos_rep = tf.repeat(tf.expand_dims(pos, 1), 
                            self.dim_len[0]*self.dim_len[1], axis=1)
        dists = tf.math.reduce_euclidean_norm(tf.cast(idxs-pos_rep, tf.float64), axis=0)
        gaussian = tf.math.exp(-tf.math.square(dists)/self.sigma)
        return gaussian


    def _get_neighbourhood(self, min_pos):
        '''
        For each sample, returns the corresponding neighbours matrix
        (the gaussian kernels centered at the corresponding
         closests' weights positions)
        '''
        pos = tf.unravel_index(min_pos, dims=[self.dim_len[0], self.dim_len[1]])
        pos = tf.transpose(pos)
        mat = tf.nest.map_structure(tf.stop_gradient, tf.map_fn(self._gaussian_kernel, pos, fn_output_signature=tf.float64))
        return mat


    def _distance_matrix(self, x):
        '''
        For each sample in x, returns distance to each weight
        ont the SOM grid
        '''
        rep_x = tf.repeat(tf.expand_dims(x, 2), 
                          self.dim_len[0]*self.dim_len[1], axis=2)
        # rep_x is B x F x D^2
        rep_w = tf.repeat(tf.expand_dims(self.w, 0), tf.shape(x)[0], axis=0)
        # rep_w is B x F x D^2
        dists = tf.math.reduce_euclidean_norm(self.w-tf.stop_gradient(rep_x),
                                              axis=1)
        return dists


    def _construct_adjmat(self, connect8):
        '''
        Used in clusters_SOM
        '''
        rows = self.dim_len[0]
        cols = self.dim_len[1]
        n = rows*cols
        # Lil matrix supports better indexing (we use sparse matrices)
        mat = lil_matrix((n, n))

        # Used to iterate over indices
        idx = np.arange(n)

        # In order to not connect adjacent pixels from different rows
        borders_rows = np.ones(n)
        borders_rows[cols-1::cols] = 0

        # 4-connectivity
        # Connect with left and right
        mat[idx[1:], idx[:-1]] = borders_rows[:-1]
        mat[idx[:-1], idx[1:]] = borders_rows[:-1]

        # Connect with up and down
        mat[idx[cols:], idx[:-cols]] = 1
        mat[idx[:-cols], idx[cols:]] = 1

        # 8-connectivity (diagonal connections)
        if connect8:
            # Down-Right
            mat[idx[:-(cols+1)], idx[(cols+1):]] = borders_rows[:-(cols+1)]
            # Up-Left
            mat[idx[(cols+1):], idx[:-(cols+1)]] = borders_rows[:-(cols+1)]
            # Down-left
            mat[idx[:-(cols-1)], idx[(cols-1):]] = borders_rows[(cols-1):]
            # Up-right
            mat[idx[(cols-1):], idx[:-(cols-1)]] = borders_rows[(cols-1):]

        return csr_matrix(mat)


    def call(self, x, training=True):
        if x.dtype != tf.float64:
            x = tf.cast(x, dtype=tf.float64)

        # x is B x F
        dists = self._distance_matrix(x)
        # dists is B x D^2
        min_pos = tf.math.argmin(dists, axis=1)
        # min_pos is B
        neig_mat = self._get_neighbourhood(min_pos)
        # neig_mat is B x D^2
        Lsom = tf.reduce_mean(neig_mat*dists)
        self.add_loss(Lsom)

        # Get closest emb and add Loss
        closest = tf.gather(self.w, indices=min_pos, axis=1)
        closest = tf.transpose(closest)
        
        if not self.get_pos:
            return closest
        
        return min_pos

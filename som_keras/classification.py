import numpy as np
import tensorflow as tf
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import gaussian_kde


# Classification methods
def classify_SOM(model, data, labels, bsize=32, normalize=False):
    '''
    Using a dataset and its labels, it assigns a class to each
    neuron by counting how many times a sample from each group
    is mapped to each of the neurons. For each neuron, the class
    with more mappings is assigned.
        data: (tf Tensor) training samples
        labels: (tf Tensor) training labels
        bsize: (int) batch size when predicting
        normalize: (bool) whether to divide each count by the amount
                    of samples of each class. Useful for 
                    unbalanced datasets.
    '''

    # Get different labels
    labels = labels.numpy().astype(int)
    labs = len(np.unique(labels))
    
    # Predict each sample
    old_getpos = model.get_pos
    model.get_pos = True
    pos = model.predict(data, batch_size=bsize).astype(int)
    
    # Count how many of each label fall in each position
    counts = np.zeros((model.dim_len[0]*model.dim_len[1], labs))
    for position, label in zip(pos, labels):
        counts[position, label] += 1
    
    # Divide by number of samples of each class
    if normalize:
        for j, lab in enumerate(np.unique(labels)):
            counts[:, j] = counts[:, j]/np.sum(labels == lab)
    
    model.counts = counts 
    
    # Get the label with more counts in each position
    grid_labs = np.argmax(counts, axis=1)
    grid_labs[np.sum(counts, axis=1)==0] = -1
    model.labels = grid_labs
    
    # Reset prediction mode
    model.get_pos = old_getpos


def classify_SOM_knn(self, data, labels, n_neighbors=1, 
                     force_calc=False, **kwargs):
    '''
    Assigns a label to each neuron of the grid using
    knn.
    data: (tf Tensor) data from which to assign labels
    labels: (tf Tensor) corresponding labels
    n_neighbors: (int) number of neighbours for the knn
    force_calc: (bool) whether to recompute labels if they have
                already been computed.
    **kwargs: arguments for KNeighborsClassifier
    '''
    if self.labels is not None and not force_calc:
        return self.labels

    knn = KNeighborsClassifier(n_neighbors=n_neighbors, 
                                **kwargs)
    knn.fit(data, labels)
    try:
        neurons = (self.w).eval()
    except:
        neurons = (self.w).numpy()
    self.labels = knn.predict(neurons.T)
    return self.labels


def classify_samples(self, data, bsize=32, get_scores=False):
    '''
    If the neurons have already been classified, it uses the SOM
    to classify a given sample(s)
        data: samples to classify
        bsize: batch size to use when predicting
        get_scores: whether to return class asignment or
                    classification score (class counts of the
                    BMU)
    returns:

    '''
    # Check that the neurons have been classified
    assert(self.labels is not None)
    old_getpos = self.get_pos
    self.get_pos = True
    positions = self.predict(data, batch_size=bsize)
    
    if not get_scores:
        predictions = self.labels[positions]
    else:
        all_scores = self.counts/np.sum(self.counts, axis=1)
        predictions = all_scores[positions]

    self.get_pos = old_getpos
    return predictions


def compute_class_prototypes(model, data_pos, labels, 
                            plot=True, **kwargs):
    '''
    Uses kde on the SOM's grid to estimate a ponderated class mean.
    model: SOM model
    data_pos: input's data predicted positions in the SOM grid
    labels: input data labels
    plot: whether to plot the prototypes or return them
    **kwargs: arguments for plot_class_prototypes
    '''
    weights = model.w.numpy()
    # Get positions in 2D
    posx, posy = np.unravel_index(data_pos, (model.dim_len[0], model.dim_len[1]))
    pos_all = np.stack((posx, posy))
    
    prototypes = np.zeros((len(np.unique(labels)), weights.shape[0]))
    
    for i, lab in enumerate(np.unique(labels)):
        # Get predictions of samples of belonging to lab
        pos = pos_all[:, labels == lab]
        
        # Fit KDE model
        kde = gaussian_kde(pos)
        
        # Get probability of each grid point
        grid = np.indices((model.dim_len[0], model.dim_len[1]))
        grid = grid.reshape(2, -1)
        probs = kde.evaluate(grid)
        probs = probs/np.sum(probs)    
       
        prototype = weights @ probs
        prototypes[i] = prototype

    if not plot:
        return prototypes
    else:
        plot_class_prototypes(prototypes, **kwargs)


# Clustering methods
def cluster_SOM(model, n_clusters=None, connect8=True, **kwargs):
    '''
    Performs hierarchical clustering on the SOM weights, only
    merging those neurons adjacent on the grid.
        model: SOM model to cluster.
        n_clusters: (int) number of clusters
        connect8: (bool) whether to allow diagonal connections
                  on the SOM grid.
        **kwargs: extra arguments for AgglomerativeClustering
    returns:
        labels: cluster labels for each position in model's grid
    '''
    adj_mat = model._construct_adjmat(connect8=connect8)
    neurons = (model.w).numpy()
    if n_clusters is not None:
        clustering = AgglomerativeClustering(n_clusters=n_clusters,
                                            connectivity=adj_mat, 
                                            **kwargs)
        labels = clustering.fit_predict(neurons.T)
    else:
        clustering = AgglomerativeClustering(connectivity=adj_mat)
        labels = clustering.fit_predict(neurons.T)
    return labels


def optimal_nclusts(model, max_clusters=10, score='davies', **kwargs):
    '''
    Computes clustering for different number of cluster and
    returns the amount with better score.
        model: SOM model to be clustered
        max_clusters: (int) maximum amount of clusters to try
        score: (str) 'davies' or 'silhouette'
        **kwargs: arguments for cluster_SOM
    returns:
        best_n (int) number of clusters with best score 
    '''
    assert(score in ['davies', 'silhouette'])
    if score == 'davies':
        score_fun = davies_bouldin_score
    else:
        score_fun = silhouette_score
    weights = model.w.numpy().T
    # Explore all clustering possibilities from 2 to max
    for i in range(2, max_clusters+1):
        clust_labs = cluster_SOM(model, i, **kwargs)
        score = score_fun(weights, clust_labs)
        if i == 2 or best_score < score:
            best_score = score
            best_n = i

    print(f'Optimal number of clusters: {best_n:.0f}')
    return best_n


def plot_cluster_dists(model, nclusts, data_pos, data_labs, 
                       lab_names=None, get_means=True, 
                       figure=None, **kwargs):
    '''
    For each cluster plots the amount of samples of each class in it.
        model: SOM model
        nclusts: number of clusters to use
        data_pos: input's data predicted positions in the SOM grid
        data_labs: input's data corresponding labels
        lab_names: names for the different labels
        get_means: whether to also compute each cluster mean
        **kwargs: arguments for plt.bar
    returns:
        means: (np array) if get_means is True returns array of
                cluster means of shape (n_clusters x n_features)
    '''

    # Predict cluster labels for SOM and data
    clust_SOM = cluster_SOM(model, nclusts)
    clust_data = clust_SOM[data_pos]

    # Vector to store each cluster mean
    weights = model.w.numpy().T
    means = np.zeros((len(np.unique(clust_SOM)), weights.shape[1]))
    if figure is None:
        figure = plt.figure(None, (35, 15))
    
    for clust in np.unique(clust_SOM):
        # Count occurences of each class
        counts = np.zeros(len(np.unique(data_labs)))
        for i, lab in enumerate(np.unique(data_labs)):
            counts[i] = np.sum(data_labs[clust_data == clust] == lab)
        
        # Bar plot
        figure.add_subplot(4, 3, clust+1)
        if lab_names is None:
            lab_names = np.unique(data_labs)
        plt.bar(lab_names, counts, **kwargs)
        plt.title(f'Cluster {clust:.0f}')
        
        # Compute cluster mean
        means[clust] = np.mean(weights[clust_SOM == clust], axis=0)
    
    if get_means:
        return means


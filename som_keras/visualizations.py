import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm
from matplotlib.colors import Normalize
from tensorflow import keras
from som_keras.classification import classify_SOM, cluster_SOM
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def visualize_labels(model, data=None, labels=None, bsize=32, 
                     normalize=False, lab_names=None, **kwargs):
    '''
    Assigns each data sample to a neuron in the SOM's 2D grid. For each position
    in the grid, displays the one with more ocurrences.
        data: (tf tensor) input data in format (n_samples x n_features)
        labels: (tf tensor) with the corresponding labels
        bsize: (int) batch size when evaluating
        normalize: (bool) whether to divide the counts by the total amount 
                    of samples of each label respectively. Recommended 
                    for unbalanced datasets.
        lab_names: (list) if not none, uses this list to construct a legend.
    returns
        f: matplotlib figure
    '''
    # If SOM grid has not been classified yet, uses input
    # to perform the classification
    if model.counts is None:
        assert(data is not None and labels is not None)
        classify_SOM(model, data, labels, bsize, normalize)

    f = plt.imshow(model.labels.reshape((model.dim_len[0], model.dim_len[1])), **kwargs)
    if lab_names is not None:
        labs = np.unique(model.labels)
        # In case some neurons do not have label
        if -1 in labs:
            lab_names = ['Not assigned']+lab_names
        
        # Get color for each label
        colors = [f.cmap(f.norm(lab)) for lab in labs]
        patches = [mpatches.Patch(color=colors[i], label=lab_names[i]) for i in range(len(labs))]
        plt.legend(handles=patches, bbox_to_anchor=(1.75, 1))
    return f


def visualize_histograms(model, data=None, labels=None, bsize=32, 
                         normalize=False, **kwargs):
    '''
    Works as visualiza_labels, but instead plots the histogram of labels
    in each position of the grid.
        data: (tf tensor) input data in format (n_samples x n_features)
        labels: (tf tensor) with the corresponding labels
        bsize: (int) batch size when evaluating
        get_counts: (bool) whether to return the ocurrences per 
                    position of the grid (True) or plot the results (False)
        normalize: (bool) whether to divide the counts by the total amount 
                    of samples of each label respectively. Recommended 
                    for unbalanced datasets.
        lab_names: (list) if not none, uses this list to construct a legend.
    returns:
        f: (matplotlib figure)
    '''
    if model.counts is None:
        assert(data is not None and labels is not None)
        classify_SOM(model, data, labels, bsize, normalize)

    labs = model.counts.shape[1]
    f = plt.figure()
    for i, count in enumerate(model.counts):
        ax = f.add_subplot(model.dim_len[0], model.dim_len[1], i+1)
        ax.bar(np.arange(labs), count, **kwargs)
        plt.xticks([])
        plt.yticks([])


def plot_dimred_pca(model, pca_data=None, connect=False, 
                        use_labels=False, get_projs=False,
                        lab_names=None, **kwargs):
    '''
    Uses PCA to represent the grid's weights in a 2D space.
        pca_data: If not None, uses this data to perform the PCA
                    Otherwise, it uses the weights.
        connect: whether to connect adjacent nodes on the grid.
        use_labels: if True and SOM neurons have been classified
                    using classify SOM, it plots each point with
                    a color corresponding to its class.
        get_projs: whether to return the projections (True)
                    or the 2D plot (False)
        **kwargs: extra arguments for PCA
    returns
        f: matplotlib figure
    '''
    weights = model.w.numpy().T
    
    # Fit PCA
    pca = PCA(n_components=2, **kwargs) 
    if pca_data is not None:
        pca.fit(pca_data)
    else:
        pca.fit(weights)
    exp_var = pca.explained_variance_ratio_
    print(f'Component 1 - Var_ratio = {exp_var[0]:.2f} , Component 2 - Var_ratio = {exp_var[1]:.2f}')
    
    # Return weights 
    projs = pca.transform(weights)
    if get_projs:
        return projs
    
    # Plot weights
    f = plt.figure()
    if not use_labels:
        plt.scatter(projs[:, 0], projs[:, 1])
    else:
        assert(model.labels is not None)
        plt.scatter(projs[:, 0], projs[:, 1], c=model.labels)
            
        # Show legend if labels are given
        if lab_names is not None:
            if len(np.unique(model.labels)) > len(lab_names):
                lab_names = ['Not assigned'] + lab_names
                norm = Normalize(vmin=-1, vmax=len(lab_names)-2)
            else:
                norm = Normalize(vmin=0, vmax=len(lab_names)-1)

            scatters = []
            for i in np.unique(model.labels):
                color = cm.viridis(norm(i))
                scat = plt.scatter(projs[model.labels == i, 0], projs[model.labels == i, 1], color=color)
                scatters.append(scat)
            plt.legend(scatters, lab_names)
    plt.xlabel(f'Principal Component 1 ({100*exp_var[0]:.0f}%)')
    plt.ylabel(f'Principal Component 2 ({100*exp_var[1]:.0f}%)')
    if connect:
        adjmat = model._construct_adjmat(False).toarray()
        for i, row in enumerate(adjmat):
            for j, entry in enumerate(row):
                if entry > 0:
                    plt.plot([projs[i, 0], projs[j, 0]], [projs[i, 1], projs[j, 1]], '-k')

    return f

def plot_dimred_tsne(model, connect=False, get_projs=False, 
                        use_labels=False, lab_names=None, **kwargs):
    '''
    Uses TSNE to represent the grid's weights in a 2D space.
        connect: whether to connect adjacent nodes on the grid.
        use_labels: if True and SOM neurons have been classified
                    using classify SOM, it plots each point with
                    a color corresponding to its class.
        get_projs: whether to return the projections (True)
                    or the 2D plot (False)
        **kwargs: extra arguments for TSNE
    returns:
        f: matplotlib figure
    '''
    weights = model.w.numpy().T
    # Project using TSNE
    tsne = TSNE(n_components=2, **kwargs) 
    projs = tsne.fit_transform(weights)
    print(f'Number of iterations: {tsne.n_iter:.0f}, Final KLD: {tsne.kl_divergence_:.4f}')
    if get_projs:
        return projs
    
    # Plot weights
    f = plt.figure()
    if not use_labels:
        plt.scatter(projs[:, 0], projs[:, 1])
    else:
        assert(model.labels is not None)
        plt.scatter(projs[:, 0], projs[:, 1], c=model.labels)
        
        # Show legend if labels are given
        if lab_names is not None:
            if len(np.unique(model.labels)) > len(lab_names):
                lab_names = ['Not assigned'] + lab_names
                norm = Normalize(vmin=-1, vmax=len(lab_names)-2)
            else:
                norm = Normalize(vmin=0, vmax=len(lab_names)-1)

            scatters = []
            for i in np.unique(model.labels):
                color = cm.viridis(norm(i))
                scat = plt.scatter(projs[model.labels == i, 0], projs[model.labels == i, 1], color=color)
                scatters.append(scat)
            plt.legend(scatters, lab_names)

    if connect:
        adjmat = model._construct_adjmat(False).toarray()
        for i, row in enumerate(adjmat):
            for j, entry in enumerate(row):
                if entry > 0:
                    plt.plot([projs[i, 0], projs[j, 0]], [projs[i, 1], projs[j, 1]], '-k')

    return f


def plot_class_prototypes(prototypes, features_dim1, features_dim2, 
                         labs_x = None, labs_y = None, 
                         plot_shape=None, lab_names=None,
                         figure=None, **kwargs):
    '''
    prototypes: class prototypes with shape (num_classes, num_features)    
    features_dim1: number of features on the x axis
    features_dim2: number of features on the y axis
    labs_x: labs to use in the x axis
    labs_y: labs to use on the y axis
    plot_shape: shape of the image representing the features
    lab_names: list of names of each label
    **kwargs: arguments for imshow
    '''
    if figure is None:
        figure = plt.figure(None, (35, 15))
    
    if plot_shape is None:
        plot_shape = [1, prototypes.shape[0]]

    for i, prototype in enumerate(prototypes):
        ax = figure.add_subplot(plot_shape[0], plot_shape[1], i+1)
        ax.imshow(prototype.reshape((features_dim1, features_dim2)), **kwargs)
        if labs_y is not None:
            ax.set_xticks(np.arange(features_dim2), minor=False)
            ax.xaxis.tick_top()
            ax.set_xticklabels(labs_y, minor=False)
                
        if labs_x is not None:
            ax.set_yticks(np.arange(features_dim1), minor=False)
            ax.set_yticklabels(labs_x, minor=False)
                
        if lab_names is not None:
            plt.title(lab_names[i])


def _grid_to_Umat(model, idxs):
    neurons_row = model.dim_len[1]
    Umat_rowlen = 2*model.dim_len[1]-1

    # Get vertical and horizontal distances
    hdist = (idxs[1]-idxs[0])%neurons_row*np.sign(idxs[1]-idxs[0])
    vdist = (idxs[1]-idxs[0])//neurons_row

    # Get idx[0]'s position on the Umat
    pos = 2*idxs[0]+(Umat_rowlen-1)*(idxs[0]//neurons_row)
    
    # Add distances to index
    Umat_idx = pos+hdist+vdist*Umat_rowlen
    
    return Umat_idx


def get_Umat(model, mean_dist=True, connect8=True):
    '''
    Builds the U-matrix corresponding to the trained grid.
    mean_dist: if True, matrix has same shape as grid and at each
                neuron, the mean distance to its neighbours is
                represented. Otherwise the matrix is bigger and 
                distances corresponding to all connections are represented
                surrounding the neurons (which have value 0)
    connect8: whether to consider diagonal connections as neighbours.
    '''
    # Get distance from each weight to the rest
    all_dists = model._distance_matrix(tf.transpose(model.w)).numpy()
    # For each neuron, get its neighbours
    adj_mat = model._construct_adjmat(connect8).toarray()
    if mean_dist:
        # Set distances to non adjacent neurons to 0
        neig_dists = all_dists*adj_mat
        # Compute mean distance
        Umat_flat = np.sum(neig_dists, axis=1)/np.sum(neig_dists > 0, axis=1)
        Umat = Umat_flat.reshape((model.dim_len[0], model.dim_len[1]))
    else:
        # Get distances only corresponding to neighbours
        pair_idx = np.indices((model.dim_len[0]*model.dim_len[1], 
                                model.dim_len[0]*model.dim_len[1]))
        pair_idx = np.transpose(pair_idx, [1, 2, 0])
        pair_idx = pair_idx[adj_mat.astype(bool)]
        neig_dists = all_dists[adj_mat.astype(bool)]
    
        # Filter repeated distances
        valid = pair_idx[:, 0] <= pair_idx[:, 1]
        neig_dists = neig_dists[valid]
        pair_idx = pair_idx[valid]

        # Translate indices from grid to U-mat indices
        neig_indices = np.apply_along_axis(_grid_to_Umat(model), 1, pair_idx)

        # Build U-matrix using distances and indices
        Umat = np.zeros((2*model.dim_len[0]-1)*(2*model.dim_len[1]-1))
        Umat[neig_indices] = neig_dists
        Umat = Umat.reshape((2*model.dim_len[0]-1, 2*model.dim_len[1]-1))
        
    return Umat


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


# Keras implementation of a Self-Organizing map.

This corresponds to a first version of a Self-Organizing map implemented as a keras.Model.

## Main features

This first version supports the following features:

+ Gaussian neighbourhood function: The function used to defined the neighbourhood corresponds to a gaussian kernel, that is, given two neurons $i, j$ with respective positions on the grid $(x_i, y_i)$, $(x_j, y_j)$, the corresponding neighbourhood function is:
$$ h(i, j) = e^{-\frac{(x_i-x_j)^2+((y_i-y_j)^2)}{\sigma}}$$

Here $\sigma$ is a hyperparameter that defined the size of the neighbourhood. Big values tend to 'include' more neurons, whilst low ones do the opposite

+ Rectangular grid: There is only support for a rectangular grid, thus at the moment it is not possible to use an hexagonal grid. The length of each of the rectangle side can be manually defined or computed proportional to the eigenvalues of the first two principal components of a PCA decomposition of the training data.

+ Initializations: Initial values can defined by uniformly sampling the first two PCA eigenvector or using random training samples.

## Examples

In the examples folder two scripts can be found:

+ train_MNIST: trains a SOM using the MNIST digits dataset and stores the resulting weights in examples/pretrained_models

+ visualize_MNIST: uses a trained version of the SOM to generate different visualizations.

## Quickstart guide

If you to train a SOM model straight on, use the following lines:

```{python}
import numpy as np
import tensorflow as tf
from tensorflow import keras
from som_keras.SOM import SOM

# Read input data
input_data = tf.convert_to_tensor(data)

# Define hyperparameters
dim_len = None # Set to None to compute grid's sides using PCA
max_dim = 32 # Max dimension for each side of the grid
sigma = 1 # Sigma of the neighbourhood function

model = SOM(dim_len=dim_len, max_dim_len=max_dim, sigma=sigma, data=input_data)

# Initialize SOM values
#model.initialize_random(data_tf)
model.initialize_with_pca(data_tf)

# Choose optimizer
optimizer = keras.optimizers.Adam(lr = 1e-3)

# Compile and train model
epochs = 300
batch_size = 128

model.compile(optimizer=optimizer, loss=None) # Loss function is built-in
model.fit(x=data_tf, y=None, batch_size=batch_size, epochs=epochs)

# Extract resulting SOM weights (n_neurons x n_features)
weights = model.get_weights()
```


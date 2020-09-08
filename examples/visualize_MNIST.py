import numpy as np
import tensorflow as tf
import pickle as pkl
import matplotlib.pyplot as plt
from os import path, mkdir
from tensorflow import keras
from som_keras.SOM import SOM
from som_keras.visualizations import *
from som_keras.classification import *

# Load hparams
save_dir = './pretrained_models/'
version = 'visual'
with open(save_dir+version+'/hparams.pkl', 'rb') as f:
    hparams = pkl.load(f)

# Load MNIST data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
data_tf = tf.random.shuffle(tf.convert_to_tensor(x_train[:20000].reshape(-1, 28*28)/255), hparams['shuffle_seed'])
labs = tf.random.shuffle(tf.convert_to_tensor(y_train[:20000]), hparams['shuffle_seed'])


# Split train-val
data_val = tf.random.shuffle(tf.convert_to_tensor(x_test[:20000].reshape(-1, 28*28)/255), hparams['shuffle_seed'])
labs_val = tf.random.shuffle(tf.convert_to_tensor(y_test[:20000]), hparams['shuffle_seed'])

# Initialize and load model
model = SOM(None, max_dim_len=hparams['max_dim'], sigma=hparams['sigma'], data=data_tf)
loss_f = keras.losses.MeanSquaredError()
optimizer = keras.optimizers.Adam(lr = 1e-3)

model.compile(optimizer=optimizer, loss=None)

check_dir = save_dir+hparams['model_name']+'/check_'+str(hparams['epochs'])+'/'
latest = tf.train.latest_checkpoint(check_dir)

model.evaluate(data_tf, batch_size=hparams['bsize'])
model.load_weights(latest)
model.evaluate(data_tf, batch_size=hparams['bsize'])

# Generate visualizations
fig_dir = './figures/'
if not path.exists(fig_dir):
    mkdir(fig_dir)

if not path.exists(fig_dir+version):
    mkdir(fig_dir+version)

# Plot final SOM weights
weights = model.w
weights = np.transpose(weights.numpy())

f = plt.figure()
for i, weight in enumerate(weights):
    x, y = np.unravel_index(i, (model.dim_len[0], model.dim_len[1]))
    f.add_subplot(model.dim_len[0], model.dim_len[1], i+1)
    ax = plt.imshow(weight.reshape((28, 28)))
    plt.xticks([])
    plt.yticks([])

plt.savefig(fig_dir+version+'/all.png')

# Plot U-matrix
plt.figure()
Umat = get_Umat(model)
plt.imshow(Umat, cmap='Greys')
plt.savefig(fig_dir+version+'/Umat.png')

# Plot label distributions at each cell
plt.figure()
visualize_histograms(model, data_tf, labs, hparams['bsize'], normalize=True)
plt.savefig(fig_dir+version+'/hists.png')

# Plot most frequent label at each cell
plt.figure()
visualize_labels(model, data_tf, labs, hparams['bsize'])
plt.savefig(fig_dir+version+'/classes.png')

preds = classify_samples(model, data_val, hparams['bsize'])
acc = np.sum(preds==labs_val.numpy())/labs_val.shape[0]
print(f'Validation accuracy: {acc}')

# Dimensionality reduction plots
plot_dimred_pca(model, connect=True, use_labels=True)
plt.savefig(fig_dir+version+'/dimredPCA.png')

plot_dimred_tsne(model, connect=True, use_labels=True)
plt.savefig(fig_dir+version+'/dimredTSNE.png')

# Plot class protypes using KDE
model.switch_preds()
data_pos = model.predict(data_tf)
prototypes = compute_class_prototypes(model, data_pos, labs)
plot_class_prototypes(prototypes, features_dim1=28, features_dim2=28,
                      plot_shape=[2, 5])
plt.savefig(fig_dir+version+'/class_prototypes')

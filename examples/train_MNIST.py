import os
import numpy as np
import pandas as pd
import pickle as pkl
import tensorflow as tf
import matplotlib.pyplot as plt
from os import path, mkdir
from tensorflow import keras
from som_keras.SOM import SOM


# First we define the different hyperparameters
hparams = {}
hparams['model_name'] = 'visual'
hparams['epochs'] = 100
hparams['sigma'] = 2
hparams['max_dim'] = 30
hparams['bsize'] = 512
hparams['shuffle_seed'] = 107312938

# Create directory to store models
save_dir = './pretrained_models/'
if not path.exists(save_dir):
    mkdir(save_dir)

if not path.exists(save_dir+hparams['model_name']):
    mkdir(save_dir+hparams['model_name'])

# Store hyperparameter dict
with open(save_dir+hparams['model_name']+'/hparams.pkl', 'wb') as f:
    pkl.dump(hparams, f) 

# Load MNIST data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
data_tf = tf.random.shuffle(tf.convert_to_tensor(x_train[:20000].reshape(-1, 28*28)/255), hparams['shuffle_seed'])
y_train = tf.random.shuffle(tf.convert_to_tensor(y_train[:20000]), hparams['shuffle_seed'])


# Initialize model
model = SOM(dim_len=None, max_dim_len=hparams['max_dim'], sigma=hparams['sigma'], data=data_tf)
#model.initialize_random(data_tf)
model.initialize_with_pca(data_tf)

# Create checkpoint directory
check_dir = save_dir+hparams['model_name']+'/check_'+str(hparams['epochs'])+'/'
if not path.exists(check_dir):
    mkdir(check_dir)
checkpoint = keras.callbacks.ModelCheckpoint(filepath=check_dir+'cp.ckpt', save_weights_only=True, 
                                             verbose=1, save_freq=50)


# Train model
optimizer = keras.optimizers.Adam(lr = 1e-3)

# Loss function is built-in, no need to specify it
model.compile(optimizer=optimizer, loss=None)
model.fit(x=data_tf, y=data_tf, batch_size=hparams['bsize'], epochs=hparams['epochs'], callbacks=[checkpoint])

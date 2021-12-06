#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 20:53:31 2021

@author: prowe

References:

https://blog.keras.io/building-autoencoders-in-keras.html

Did I use this?

For general NN with keras:
https://victorzhou.com/blog/keras-neural-network-tutorial/
    
For validation:
https://towardsdatascience.com/addressing-the-difference-between-keras-validation-split-and-sklearn-s-train-test-split-a3fb803b733
"""


import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense
#from tensorflow.keras.callbacks import EarlyStopping
#from tensorflow.keras.utils import to_categorical
#from tensorflow.keras.optimizers import Adam

import keras
from keras import layers



def tune_model(x_train, x_test, loss_funs, epochs, batch_sizes):
    """
    Create figures showing the loss and MAE for different hyperparameters
    @param x_train The training data
    @param x_test The testing data
    @param loss_funs List of strings of loss functions to test
    @param epochs Number of epochs to try
    @param batch_sizes Batch sizes to try
    """
    for iloss, loss_fun in enumerate(loss_funs):    
        autoencoder.compile(optimizer = 'adam', loss = loss_fun)
        losses = np.zeros((epochs, len(batch_sizes)))
        mse = np.zeros(len(batch_sizes))
        for i,batch_size in enumerate(batch_sizes):
            losses[:,i] = train_model(x_train, x_test, epochs, batch_size)
            mse[i] = test_model(x_test)

        # Plot the losses with epochs for different batch sizes, for loss fun
        plt.figure(iloss+1)
        plt.cla()
        evec = np.arange(epochs)
        for i,b in enumerate(batch_sizes):
            plt.plot(evec, losses[:,i], 'o-', label = 'batch size: ' + str(b))
        plt.title('loss function:' + loss_fun)
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend()
        
        # Plot the mse
        plt.figure(10)
        plt.plot(batch_sizes, mse, label = loss_fun)
        
    plt.xlabel('batch size')
    plt.ylabel('MSE')
    plt.figure(10)
    plt.legend()
    
    
def train_model(x_train, x_test, epochs, batch_size):
    """
    Train the model for a given set of hyperparameters
    @param x_train The training data
    @param x_test The testing data
    @param epochs The number of epochs
    @param batch_size The batch size
    @return The loss function
    """
    
    history = autoencoder.fit(x_train, x_train, epochs = epochs,
                              batch_size = batch_size, shuffle = True,
                              verbose = 0, validation_data = (x_test, x_test))
    # The following does not work. Gives error message:
    # "IndexError: list index out of range "
    # autoencoder.evaluate(x_test)
    
    return history.history['loss']

def test_model(x_test):
    """
    Test the model accuracy by encode and decoding inputs from the test set.
    I don't know how to get the accuracy on the test set, so I computed the MAE
    @param x_test The test set
    @return MAE the mean absolute difference
    """
    x_encoded = encoder.predict(x_test)
    x_decoded = decoder.predict(x_encoded)
    
    return np.mean(np.abs(x_test - x_decoded)) 
    


# # # # # # # # # #      Read in the data      # # # # # # # # # #
# Flags for the run
plot_figures = False

# The input file
INPUT_FILE = '../data/frotz_builtin_walkthrough.csv'

# Read in the data
df = pd.read_csv(INPUT_FILE)
observations = (df['Observation']).values    # The inputs
actions = (df['Action']).values              # The labels, or correct answers
points = (df['Points']).values               # The points earned by the action

# # # # # # # # # #      Clean and set up the data      # # # # # # # # # #
# Encode the observations
# The inputs are size: inputs x features: (396 x 768)
# The inputs vary from -1.2 to 1.4, so they are already pretty well
# centered and scaled, and they are already flat
# This encoder produces vectors of length 768: multi-qa-mpnet-base-dot-v1
# This one produces vectors of length 3xx: all-MiniLM-L6-v2
sentenceTransformer = SentenceTransformer('multi-qa-mpnet-base-dot-v1')
inputs = np.array([sentenceTransformer.encode([x])[0] for x in observations])

# Note that we DO NOT normalize the data to values between 0 and 1

# Get the lengths of our variables
n_inputs = np.shape(inputs)[0]
n_features = np.shape(inputs)[1]
n_outputs = n_inputs     # length of one-hot vectors for output

# Get the labels as indices to the list of unique actions
# The labels are (n_inputs,) or (396,)
outputs = inputs


# Setup
# encoding_dim is the size of our encoded representations
# Input() is used to instantiate a Keras tensor.
# encoded is the encoded representation of the input
# decoded is the lossy reconstruction of the input
# autoencoder maps an input to its reconstruction
# encoder maps an input to its encoded representation
# encoded_input is our encoded (32-dimensional) input
# decoder_layer retrieve sthe last layer of the autoencoder model
# decoder is the decoder model
encoding_dim = 50  # floats -> compression of x, assuming input is x floats
input_obs = keras.Input(shape=(n_features,))
encoded = layers.Dense(encoding_dim, activation='tanh')(input_obs)
decoded = layers.Dense(n_features, activation='tanh')(encoded)
autoencoder = keras.Model(input_obs, decoded)
encoder = keras.Model(input_obs, encoded)
encoded_input = keras.Input(shape=(encoding_dim,))
decoder_layer = autoencoder.layers[-1]
decoder = keras.Model(encoded_input, decoder_layer(encoded_input))

# Divide into training and testing sets
x_train, x_test, _, _ = train_test_split(inputs, inputs,  test_size = 0.2,
                                         random_state = 1)

print('Training inputs shape:', x_train.shape)
print('Testing inputs shape:', x_test.shape)


# Hyperparameter tuning
loss_funs = ['binary_crossentropy', 'MSE']
epochs = 100
batch_sizes = [100, 150, 200, 250, 300, np.shape(x_train)[0]]
#tune_model(x_train, x_test, loss_funs, epochs, batch_sizes)
    

# Select the best hyperparameters from above. Should I run the model for all 
# inputs here, or use the train/test set again?
loss_fun = 'MSE'
epochs = 50
batch_size = np.shape(x_train)[0]
autoencoder.compile(optimizer = 'adam', loss = loss_fun)
history = autoencoder.fit(x_train, x_train, epochs = epochs, 
                          batch_size = batch_size,  shuffle = True, 
                          validation_data = (x_test, x_test))
x_encoded = encoder.predict(inputs)
x_decoded = decoder.predict(x_encoded)    

#TODO make this into a method in a class and put in an external file to be
# imported so Diggy can use it too
diff = np.zeros((np.shape(inputs)))
mindiff = np.zeros(n_inputs)
for i,x in enumerate(x_decoded):
    for j in range(encoded):
        diff[j] = sum(np.abs(x[:,j] - x_train[:,i]))
    mindiff[i] = min(diff)  #mindiff of i should be i


# # In the following, the reload does not work:
# encoder.save_weights("encoder_weights.h5")
# reconstructed_encoder = keras.models.load_model("model.h5")


# # Try loading in and using the model on the walkthrough
# # autoencoder.save_weights('model.h5')
# autoencoder.save('model.h5')
# reconstructed_model = keras.models.load_model("model.h5")


if plot_figures:
    # Plot the inputs to see what they look like
    ifig = 4
    plt.figure(ifig); plt.clf()
    plt.plot(np.arange(np.shape(inputs)[0]), inputs)
    plt.xlabel('observation')
    plt.ylabel('value')
    plt.title('Each curve represents the values for a particular feature')
    
    ifig += 1
    plt.figure(ifig); plt.clf()
    plt.plot(np.arange(np.shape(inputs)[1]), inputs.T)
    plt.xlabel('feature')
    plt.ylabel('value')
    plt.title('Each curve represents the values for a particular observation')
    
    # Plot the encoded data, just to see what it looks like
    ifig += 1
    plt.figure(ifig); plt.clf()
    plt.plot(np.arange(np.shape(x_encoded)[0]), x_encoded)
    plt.xlabel('observation')
    plt.ylabel('value')
    plt.title('Each curve represents the values for a particular feature')
    
    ifig += 1
    plt.figure(ifig); plt.clf()
    plt.plot(np.arange(np.shape(x_encoded)[1]), x_encoded.T, '.-')
    plt.xlabel('feature')
    plt.ylabel('value')
    plt.title('Each curve represents the values for a particular observation')
    
    
    # Plot a particular test case at index i
    i = 0
    ifig += 1
    plt.figure(ifig); plt.clf()
    plt.subplot(211)
    plt.plot(inputs[i],label = 'Original')
    plt.plot(x_decoded[i], label = 'Decoded')
    plt.legend()
    plt.subplot(212)
    plt.plot(inputs[i] - x_decoded[i], label = 'original - decoded')
    plt.legend()
    
    ifig += 1
    plt.figure(ifig)
    plt.title('Original - Decoded')
    for i in range(np.shape(x_test)[0]):
        plt.plot(x_test[i] - x_decoded[i])
    
# Question for Prof. Chambers:
# Can I throw out the features for which the encoded values are all zero?
    
    
    
    

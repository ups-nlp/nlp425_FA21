#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 20:53:31 2021

@author: prowe

References:

For general NN with keras:
https://victorzhou.com/blog/keras-neural-network-tutorial/
    
For validation:
https://towardsdatascience.com/addressing-the-difference-between-keras-validation-split-and-sklearn-s-train-test-split-a3fb803b733
"""


import sys
if '../' not in sys.path:
    sys.path.append('../')

from pca_encoder import PCAencoder

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import models
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import pickle
#from sentence_transformers import SentenceTransformer


def build_model(epochs, batch_size, hidlay1_nodes, hidlay2_nodes, learnrate):
    """Build neural network model with up to 2 hidden layers"""
    # Build the model: the activation functions were relu for the first two
    # and softmax for the last, now using 'tanh' for all
    model = Sequential()
    actfun = 'sigmoid'
    
    # Add layers for 2 hidden layers, 1 hidden layer, or no hidden layers
    if hidlay2_nodes > 0:
        model.add(Dense(hidlay1_nodes, activation=actfun, input_shape=(n_features,)))
        model.add(Dense(hidlay2_nodes, activation=actfun))
        model.add(Dense(n_actions, activation = 'softmax'))
    elif hidlay1_nodes > 0:        
        model.add(Dense(hidlay1_nodes, activation=actfun, input_shape=(n_features,)))
        model.add(Dense(n_actions, activation = 'softmax'))
    else:
        model.add(Dense(n_actions, activation='softmax', input_shape=(n_features,)))
        
        
    # Compile the model. The loss function was categorical_crossentropy, now
    # using MSE
    model.compile(optimizer = Adam(lr = learnrate),
                  loss = 'MSE',
                  metrics=['accuracy'])
    
    # Train the model. The use of to_categorical converts the action-indices to 
    # one-hot vectors
    history = model.fit(train_obs,
                        train_labels,
                        verbose = 0,
                        validation_split = 0.2, # split data in 80/20 sets
                        epochs = epochs,
                        batch_size = batch_size)

    loss, accuracy = model.evaluate(test_obs, test_labels)
    
    return loss, accuracy, history, model


# # # # # # # # # #      Read in the data      # # # # # # # # # #
INPUT_FILE = '../data/frotz_builtin_walkthrough.csv'
df = pd.read_csv(INPUT_FILE)
observations = (df['Observation']).values    # The inputs
actions = (df['Action']).values              # The labels, or correct answers
points = (df['Points']).values               # The points earned by the action


# Encode the observations
pcaEncoder = PCAencoder(observations)
obs = pcaEncoder.proj_down

# Get a unique list of actions 
for i,action in enumerate(actions):
    actions[i] = action.lower()
unique_actions = list(set(actions))

# Get the lengths of our variables
n_inputs = np.shape(obs)[0]
n_features = np.shape(obs)[1]
n_actions = len(unique_actions)      # length of one-hot vectors for output

# Get the labels as indices to the list of unique actions
# The labels are (n_inputs,) or (396,)
labels = np.zeros((n_inputs, n_actions))
for iact,action in enumerate(actions):
    labels[iact,unique_actions.index(action)] = 1


# Divide into training and testing sets
train_obs, test_obs, \
train_labels, test_labels = train_test_split(obs, labels, 
                                             test_size = 0.2,
                                             random_state = 1)


# # # # # # # # # #      Build and run the model      # # # # # # # # # #
# Tuned Hyperparameters (see ee_neural_network_tuning)
epochs = 50
batch_size = 10
hidden_layer1_nodes = 50
hidden_layer2_nodes = 0
learning_rate = 0.005


model = Sequential()
model.add(Dense(hidden_layer1_nodes, activation = 'sigmoid', 
                input_shape=(n_features,)))
model.add(Dense(n_actions, activation = 'softmax'))
    
    
# Compile the model. The loss function was categorical_crossentropy, now
# using MSE
model.compile(optimizer = Adam(lr = learning_rate),
              loss = 'MSE',
              metrics=['accuracy'])

# Train the model. The use of to_categorical converts the action-indices to 
# one-hot vectors
history = model.fit(train_obs,
                    train_labels,
                    verbose = 0,
                    validation_split = 0.2, # split data in 80/20 sets
                    epochs = epochs,
                    batch_size = batch_size)

loss, accuracy = model.evaluate(test_obs, test_labels)



predictions = model.predict(train_obs)

# The predictions are all very similar to each other and close to 0
plt.figure()
plt.plot(predictions)

plt.figure()
plt.plot(predictions.T)

# # Evaluate the model.
model.evaluate(test_obs, test_labels)

# Save the model and the unique actions
model.save('ee_neural_network_model.pb')
with open('unique_actions.pkl', 'wb') as fid:
    pickle.dump(unique_actions, fid)

# Load in and try out
new_model = models.load_model('ee_neural_network_model.pb')
predict2 = new_model.predict(train_obs)
plt.figure()
plt.plot(predictions-predict2)


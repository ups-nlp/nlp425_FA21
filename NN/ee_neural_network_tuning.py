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
# Hyperparameters
epochs = 100
batch_size = 50
hidden_layer1_nodes = 20
hidden_layer2_nodes = 0
learnrate = 0.005



# # # # # # # # # #      Test batch sizes and epochs      # # # # # # # # # #
epochses = [50, 100]
batch_sizes = [10, 50, 200, 300, 316]
plt.figure(1)
plt.clf()
losses = np.zeros((len(epochses),len(batch_sizes)))
accuracies = np.zeros((len(epochses),len(batch_sizes)))
for i,epochs in enumerate(epochses):
    for j,batch_size in enumerate(batch_sizes):
        loss, accuracy, history, _ = build_model(epochs, batch_size, 
                                                 hidden_layer1_nodes, 
                                                 hidden_layer2_nodes, 
                                                 learnrate)
        losses[i,j] = loss
        accuracies[i,j] = accuracy
        
        
        # Plot the accuracy of the model as it trains
        lab_str = 'batch size: ' + str(batch_size) + ', epoch: ' + str(epochs)
        plt.plot(history.history['accuracy'], label=lab_str)

plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()

plt.figure(2)
plt.clf()
plt.plot(batch_sizes, accuracies.T, 'o-')
plt.legend(epochses)
plt.xlabel('Batch size')
plt.ylabel('Accuracy')
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #




# # # # # # #      Test number of hidden layers and nodes    # # # # # # # #
epochs = 50
batch_size = 10 #50
learnrate = 0.005

hidden_layer1_nodes_list = [0, 20, 50, 100, 200, 300] #30, 30]
hidden_layer2_nodes_list = [0,  0,  0,  0,    0,   0] #80]

plt.figure(3)
plt.clf()
losses = []
accuracies = []
for i,hidden_layer1_nodes in enumerate(hidden_layer1_nodes_list):
    hidden_layer2_nodes = hidden_layer2_nodes_list[i]
    loss, accuracy, history, _ = build_model(epochs, batch_size, 
                                             hidden_layer1_nodes, 
                                             hidden_layer2_nodes, 
                                             learnrate)
    losses.append(loss)
    accuracies.append(accuracy)
    
    # Plot the accuracy of the model as it trains
    label_str = 'nodes in hidden layers: ' + str(hidden_layer1_nodes) \
                + ', ' + str(hidden_layer2_nodes)
    plt.plot(history.history['accuracy'], label = label_str)

plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()

plt.figure(5)
plt.clf()
plt.plot(hidden_layer1_nodes_list, accuracies, 'o-')
plt.xlabel('Nodes in hidden layer')
plt.ylabel('Accuracy')



# Repeat but with 2 hidden layers
hidden_layer1_nodes_list = [20,  50, 100, 200, 300] 
hidden_layer2_nodes_list = [50, 100, 200, 300, 300] 

losses = []
accuracies = []
plt.figure(3)
for i,hidden_layer1_nodes in enumerate(hidden_layer1_nodes_list):
    hidden_layer2_nodes = hidden_layer2_nodes_list[i]
    loss, accuracy, history, _ = build_model(epochs, batch_size, 
                                             hidden_layer1_nodes, 
                                             hidden_layer2_nodes, 
                                             learnrate)
    losses.append(loss)
    accuracies.append(accuracy)
    
    # Plot the accuracy of the model as it trains
    label_str = 'nodes in hidden layers: ' + str(hidden_layer1_nodes) \
                + ', ' + str(hidden_layer2_nodes)
    plt.plot(history.history['accuracy'], label = label_str)

plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()

plt.figure(5)
plt.plot(hidden_layer1_nodes_list, accuracies, 'o-')
plt.xlabel('Nodes in hidden layer')
plt.ylabel('Accuracy')
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


#plt.figure(3)
#plt.clf()
#plt.plot(batch_sizes, losses, 'o-')


# Tuned Hyperparameters
epochs = 50
batch_size = 10
hidden_layer1_nodes = 50
hidden_layer2_nodes = 100
learnrate = 0.005

loss, accuracy, history, model = build_model(epochs, batch_size, 
                                             30, 20, learnrate)


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

# Overview:
# - The model accuracy is too poor to be useable (~20%)
# - Tuning of hyperparameters including batch_size, epochs, number of
#   hidden layers, and nodes in hidden layers made no improvement.
# - The prediction on the *training* set is extremely poor (~14%), and
#   strangely much poorer than the accuracy suggested
# - The predictions themselves are very poor - all results look approx.
#   the same, and all values are close to zero, with maxima around 0.1


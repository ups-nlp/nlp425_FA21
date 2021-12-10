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


# # # # # # # # # #      Read in the data      # # # # # # # # # #
INPUT_FILE = '../data/frotz_builtin_walkthrough.csv'
df = pd.read_csv(INPUT_FILE)
observations = (df['Observation']).values    # The inputs
actions = (df['Action']).values              # The goal outputs 
points = (df['Points']).values               # The points earned by the action


# Encode the observations
pcaEncoder = PCAencoder(observations)
obs = pcaEncoder.proj_down

# Get a unique list of actions 
#for i,action in enumerate(actions):
#    actions[i] = action.lower()
actions = [x.lower() for x in actions]
unique_actions = list(set(actions))

# Get the lengths of our variables
n_inputs = np.shape(obs)[0]
n_features = np.shape(obs)[1]
n_actions = len(unique_actions)      # length of one-hot vectors for output

# Get the one-hot labels
onehot_labels = np.zeros((n_inputs, n_actions))
for iact,action in enumerate(actions):
    onehot_labels[iact, unique_actions.index(action)] = 1


# Divide into training and testing sets
train_obs, test_obs, \
train_labels, test_labels = train_test_split(obs, onehot_labels, 
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
model.add(Dense(hidden_layer1_nodes, activation = 'tanh', 
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
plt.xlabel('Observation')
plt.ylabel('Probability')
plt.title('Probability determined by NN for Actions')

def plot_result(i):
    plt.plot(predictions[i,:], label=str(np.argmax(predictions[i,:]))+','+str(train_labels[i,:].argmax()))

plt.figure()
plt.subplot(4,1,1)
plot_result(0)
plot_result(2)
plot_result(3)
plt.legend()
plt.title('Probability determined by NN for Observations')

plt.subplot(4,1,2)
plot_result(4)
plot_result(5)
plot_result(6)
plt.legend()
plt.ylabel('Probability')

plt.subplot(4,1,3)
plot_result(7)
plot_result(8)
plot_result(9)
plt.legend()

plt.subplot(4,1,4)
plot_result(10)
plot_result(11)
plot_result(12)
plt.legend()

plt.xlabel('Action')

# Evaluate the model.
model.evaluate(test_obs, test_labels)


# # # # #   SAVE THE RESULTS   # # # # # # #
# Save the model and the unique actions
model.save('ee_neural_network_model.pb')
with open('unique_actions.pkl', 'wb') as fid:
    pickle.dump(unique_actions, fid)

# Save the labels as indices to the list of unique actions
# The labels are (n_inputs,) or (396,)
np.savetxt('../data/action_index.csv', 
           [unique_actions.index(x) for x in actions],
           delimiter = ',')
# # # # # # # # # # # # # # # # # # # # # #



# # # # #   Load in and try out   # # # # # 
new_model = models.load_model('ee_neural_network_model.pb')
predict2 = new_model.predict(train_obs)

print('Predictions agree before after loading in model to within',
      np.sum(predictions-predict2))
# # # # # # # # # # # # # # # # # # # # # #



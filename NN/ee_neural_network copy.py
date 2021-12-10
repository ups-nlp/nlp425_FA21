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
from sentence_transformers import SentenceTransformer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

# # # # # # # # # #      Read in the data      # # # # # # # # # #
INPUT_FILE = '../data/frotz_builtin_walkthrough.csv'
df = pd.read_csv(INPUT_FILE)
observations = (df['Observation']).values    # The inputs
actions = (df['Action']).values              # The labels, or correct answers
points = (df['Points']).values               # The points earned by the action

# # # # # # # # # #      Clean and set up the data      # # # # # # # # # #
# Encode the observations
# The inputs are size: inputs x features: (396 x 768)
# The inputs vary from -1.2 to 1.4, so they are already pretty well
# centered and scaled, and they are already flat
sentenceTransformer = SentenceTransformer('multi-qa-mpnet-base-dot-v1')
obs = np.array([sentenceTransformer.encode([x])[0] for x in observations])

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
labels = np.zeros((n_inputs))
for i,action in enumerate(actions):
    labels[i] = unique_actions.index(action)


# Divide into training and testing sets
train_obs, test_obs, \
train_labels, test_labels = train_test_split(obs, labels, test_size = 0.2,
                                             random_state = 1)


# # # # # # # # # #      Build and run the model      # # # # # # # # # #
# Hyperparameters
epochs = 20
batch_size = 32
hidden_lyr1_nodes = 64
hidden_lyr2_nodes = 64
learning_rate = 0.005

# Build the model
model = Sequential([
  Dense(hidden_lyr1_nodes, activation='relu', input_shape=(n_features,)),
  Dense(hidden_lyr2_nodes, activation='relu'),
  Dense(n_actions, activation='softmax'),
])

# Compile the model.
model.compile(
  optimizer=Adam(lr=learning_rate),
  loss='categorical_crossentropy',
  metrics=['accuracy'],
)

# Train the model. The use of to_categorical converts the indices to the
# actions to one-hot vectors
history = model.fit(train_obs,
                    to_categorical(train_labels),
                    verbose = 0,
                    validation_split = 0.2, # split data in 80/20 sets
                    epochs = epochs,
                    batch_size = batch_size)


# Plot the accuracy of the model as it trains
plt.plot(history.history['accuracy'])
plt.xlabel('epochs')
plt.ylabel('accuracy')

# Evaluate the model.
# output: accuracy is 0.425 on test set
model.evaluate(test_obs, to_categorical(test_labels))



# # Save the model to disk.
# model.save_weights('model.h5')

# # Load the model from disk later using:
# # model.load_weights('model.h5')

# # Predict on the first 5 test images.
# predictions = model.predict(test_images[:5])

# # Print our model's predictions.
# print(np.argmax(predictions, axis=1)) # [7, 2, 1, 0, 4]

# # Check our predictions against the ground truths.
# print(test_labels[:5]) # [7, 2, 1, 0, 4]
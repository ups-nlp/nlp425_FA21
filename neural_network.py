"""
Neural network for text-based game walkthrough

author @prowe
Nov. 17, 2021
"""
# Installed modules
import tensorflow as tf
from sentence_transformers import SentenceTransformer

# Built-in modules
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

# Our modules
from nn_resources import train_model




# Adapted from:
#   https://www.pyimagesearch.com/2020/03/23/
#   using-tensorflow-and-gradienttape-to-train-a-keras-model/



# Train the neural network using the walk-through


# # # # # # # # # #      Read in the data      # # # # # # # # # #
INPUT_FILE = 'data/frotz_builtin_walkthrough.csv'
df = pd.read_csv(INPUT_FILE)
observations = (df['Observation']).values    # The inputs
actions = (df['Action']).values              # The labels, or correct answers
points = (df['Points']).values               # The points earned by the action

# Encode the observations
sentenceTransformer = SentenceTransformer('multi-qa-mpnet-base-dot-v1')
x = np.array([sentenceTransformer.encode([x])[0] for x in observations])


# Convert the actions into a unique list of actions and an index to
# the list itemp
for i,action in enumerate(actions):
    actions[i] = action.lower()
unique_actions = list(set(actions))

y = np.zeros(len(x))
for i,action in enumerate(actions):
    y[i] = unique_actions.index(action)

# Make labels between 0 and 1
y = y / max(y)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #



# # # # # #      Neural network parameters      # # # # # # # # # #
HIDDEN_LAYERS = 0       # Can only be 0 or 1 for now
HIDDEN_NODES = 9        # Only used if hidden_layers = 1
NOUT = 1                # Number of output variables
nfeat = np.shape(x)[1]  # number of features



if HIDDEN_LAYERS == 0:
    W1 = tf.Variable(tf.random.normal([nfeat, NOUT], stddev=0.03), name='W1')
    b1 = tf.Variable(tf.random.normal([NOUT]), name='b1')
    Ws = [W1]
    bs = [b1]
elif HIDDEN_LAYERS == 1:
    # Declare the weights connecting the input to the hidden layer
    W1 = tf.Variable(tf.random.normal([nfeat, HIDDEN_NODES], stddev=0.03), name='W1')
    b1 = tf.Variable(tf.random.normal([HIDDEN_NODES]), name='b1')
    # and the weights connecting the hidden layer to the output layer
    W2 = tf.Variable(tf.random.normal([HIDDEN_NODES, NOUT], stddev=0.03), name='W2')
    b2 = tf.Variable(tf.random.normal([1]), name='b2')
    Ws = [W1] + [W2]
    bs = [b1] + [b2]
else:
    raise ValueError('Only 0 or 1 hidden layers implemented.')




# We aren't going to have a training and testing set, because the test
indices = list(range(len(x)))
NTRAIN = np.shape(y)[0]
BATCH_SIZE = 10 #NTRAIN
EPOCHS = 2 * BATCH_SIZE
INIT_LR = 0.5

loss, prediction = train_model(x, y, EPOCHS, BATCH_SIZE, INIT_LR,  Ws, bs)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 3 12:19:30 2021

@author: Eric Markewitz
"""

import sys
sys.path.append('../')
#print(sys.path)
from dep_agent import DEPagent


#Decisionmaker NEURAL NET
trainingInputs = []
trainingOutputs = []

#file containing the observation, action, and module of that action from the walkthrough
dm_training_data = open("../data/dm_training_data.txt")



for line in dm_training_data:
    list = line.split(',')
    observation = list[0]
    obs_vect = create_vect(observation)

    action = list[1]
    module = list[2]
    module = re.sub('\n', '', module)

    trainingInputs.append(obs_vect)
    trainingOutputs.append(module)


np_input = np.array(trainingInputs)
np_output = np.array(trainingOutputs)



train_obs, test_obs, train_labels, test_labels = train_test_split(np_input, np_output, test_size = 0.2, random_state =1)

#Hyperparameters
epochs = 20
batch_size = 32
hidden_lyr1_nodes = 64
hidden_lyr2_nodes = 64
learning_rate = 0.005
input_size = 50
output_size = 4

# Build the model
model = Sequential([
  Dense(hidden_lyr1_nodes, activation='relu', input_shape=(input_size,)),
  Dense(hidden_lyr2_nodes, activation='relu'),
  Dense(output_size, activation='softmax'),
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
                    epochs=epochs,
                    batch_size=batch_size)


# Plot the accuracy of the model as it trains
plt.plot(history.history['accuracy'])
plt.xlabel('epochs')
plt.ylabel('accuracy')

# Evaluate the model.
# output: accuracy is 0.425 on test set
model.evaluate(test_obs, to_categorical(test_labels))

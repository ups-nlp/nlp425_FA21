#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 18:59:14 2021

@author: Penny Rowe, Eric, Danielle Dolan
"""

# Built-in modules
import random
import numpy as np
import re

# Installed modules
from jericho import FrotzEnv
from numpy import dot
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

from operator import add
from operator import truediv

# In-house modules
from agent import Agent



class DEPagent(Agent):
    """Agent created by Danielle, Eric, and Penny."""

    def __init__(self):
        """ Initialze the class by setting instance variables for enemies,
        weapons, and movements """

        # Lists of enemies and weapons
        # Perhaps these should use intelligence to build up?
        # For now, just add them manually
        self.enemies = ['troll', 'trolls', 'thief','thieves', 'grue', 'grues']
        self.weapons = ['sword', 'knife', 'axe']

        # Movements and their opposites
        self.movements = {'north':'south', 'south':'north',
                          'east':'west', 'west':'east',
                          'up':'down', 'down':'up',
                          'northwest':'southeast', 'southeast':'northwest',
                          'northeast':'southwest', 'southwest':'southeast', 'go':'go'}

        # index of the current observation in the enviroment
        self.OBSERVATION_INDEX = 8

        # Number of past action to check
        self.PAST_ACTIONS_CHECK = 3

        self.vocab_vectors, self.word2id = self.embed_vocab()

        #NEURAL NET STUFF HERE
        trainingInputs = []
        trainingOutputs = []
        dm_training_data = open("./data/dm_training_data.txt")
        for line in dm_training_data:
            list = line.split(',')
            observation = list[0]
            obs_vect = self.create_vect(observation)

            action = list[1]
            module = list[2]
            module = re.sub('\n', '', module)

            trainingInputs.append(obs_vect)
            trainingOutputs.append(module)

            trainingData.append(dataLine)

        for line in trainingData:

            print(line)
        '''
        train_obs, test_obs, train_labels, test_labels = train_test_split(obs, labels, test_size = 0.2, random_state =1)

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
        '''

        # set model for sentance transformers
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def hoarder(self, env:FrotzEnv, valid_actions:list, history:list) -> str:
        """
        Determine what action the hoarder would take.

        @param valid_actions
        @param history

        For the moment, hoarder will return "take all" until futher
        intelligence can be added.

        @return chosen_action: A string of containing "take all"

        """
        #return random.choice(valid_actions)
        return self.get_action(env, valid_actions, history)


    def fighter(self, env:FrotzEnv, valid_actions:list, history:list) -> str:
        """
        @param env: The environment variable, which contains the inventory
        @param valid_actions: The list of valid actions
        @param history: A list of tuples (observation, action)

        For the moment: fighter will return "kill ___ with ____" if an enemy
        and a weapon are identified. Otherwise it will run.

        @return chosen_action: String containing "kill ____ with ____" or "run"
        """
        #return random.choice(valid_actions)
        return self.get_action(env, valid_actions, history)

        #inventory = [item.name for item in env.get_inventory()]
        #for action in valid_actions:
         #   have_enemy = [enemy in action for enemy in self.enemies]
         #   if any(have_enemy):
        """
                enemy = self.enemies[have_enemy.index(True)]
                # We are facing an enemy: if we have a weaon, fight with it
                # if not, run
                have_weapons = [weapon in inventory for weapon in self.weapons]
                if any(have_weapons):
                    weapon = self.weapons[have_weapons.index(True)]
                    return 'kill ' + enemy + ' with ' + weapon
                return 'run'
        return 'run'
        """

    def mover(self, env:FrotzEnv, valid_actions:list, history:list) -> str:
        """
        @param valid_actions
        @param history

        The mover will take some set of valid move actions, cross that will
        other known move actions and will pick one by prioritizing new
        directions

        @return chosen_action: A string containing a move action
        """

        if not any(valid_actions):
            return 'no valid actions'

        if len(valid_actions) == 1:
            return valid_actions[0]

        valid_movements = [action for action in self.movements \
                             if action in valid_actions]
        if not any(valid_movements):
            return random.choice(valid_actions)


        # Check the history for the last movement and don't go in the
        # opposite direction
        i = 0
        last_movement = []
        while not any(last_movement) and i < len(history):
            i += 1
            last_movement = [action for action in self.movements \
                             if action in history[-i][1] ]

        # last_movement should typically have only one element, unless the
        # last action was something like 'go north then south' in which case
        # this will not work very well
        if any(last_movement):
            double_back = self.movements[last_movement[0]]
            if double_back in valid_movements and len(valid_movements)>1:
                valid_movements.remove(double_back)

        # Pick randomly from remaining choices
        #return random.choice(valid_movements)
        return self.get_action(env, valid_movements, history)


    def everything_else(self, env: FrotzEnv, valid_actions:list, \
                        history:list) -> str:
        """
        Feed the observation and list of vaild actions into a model
        (similarity / neural network) that will decide what the next
        action will be

        @param valid_actions
        @param history

        @return chosen_action: A String containing a new action
        """
        # get list of past actions
        """
        if len(history) != 0:
            past_actions = []
            for combo in history:
                past_actions.append(combo[1])
        observation = env.get_state()[8]

        # Encode the observation
        query_vec = self.model.encode([observation])[0]


        # set up for testing all actions
        chosen_action = ""
        best_similarity = 0
        # for each vaild action, test its similarity to the observation
        for action in valid_actions:
            sim = dot(query_vec, self.model.encode([action])[0]) \
                      /(norm(query_vec)*norm(self.model.encode([action])[0]))
            # choose action with the best similarity
            if (best_similarity < sim):
                if (len(history) !=0 and past_actions.count(action)>1):
                    continue
                best_similarity = sim
                chosen_action = action
        print(" similarity: ", best_similarity)

        """
        # Or ... run the neural network: The input is the observation, which
        # has been encoded as query_vec. The output is the result after
        # running through the NN


        #return chosen_action    # action with the best similarity to the observation
        return self.get_action(env, valid_actions, history)


    def take_action(self, env: FrotzEnv, history: list) -> str:
        """
        Takes in the history and returns the next action to take

        @param env, Information about the game state from Frotz
        @param history, A list of tuples of previous actions and observations

        @return action, A string with the action to take
        """

        # Eventually the valid actions will be determined by Team 1,
        # but for now use the FrotzEnv
        valid_actions = env.get_valid_actions()

        # get sorted actions: in order: Hoarder, mover, fighter, and everything else.
        sorted_actions = self.sort_actions(valid_actions)

        chosen_module = self.decision_maker(valid_actions, history, env)

        action_modules = [self.hoarder,
                          self.mover,
                          self.fighter,
                          self.everything_else]

        # accounting for empty action lists, if empty, pick mover module
        if len(sorted_actions[0]) == 0 and chosen_module == 0:
            chosen_module = 1
        elif len(sorted_actions[2]) == 0 and chosen_module == 2:
            chosen_module = 1
        elif len(sorted_actions[3]) == 0 and chosen_module == 3:
             chosen_module= 1

        action = action_modules[chosen_module](env, sorted_actions[chosen_module], history)
        return action


    def decision_maker(self, valid_actions:list, history:list, env: FrotzEnv) -> int:
        """
        Decide which choice to take randomly for now
        this needs some intelligence

        @param valid_actions
        @param history
        @param environment

        Creates an embedding of all the words in the previous observation,
        runs that through a neural network that ranks how much we should use
        each of the modules. Then returns an int that represents the module
        with the highest value that has valid actions

        @return chosen_module: an integer between 0 and 3, representing the
                               action module we are going to use
        """

        vector = self.create_observation_vect(env)

        sorted_actions = self.sort_actions(valid_actions)

        chosen_module = random.randint(0, 3)
        return chosen_module


    def embed_vocab(self) -> (list, dict):
        """
        Reads in the vocab and vector GloVemaster files in from the data folder.
        Returns a dictionary matching a word to an index and a list of the vectors

        @return list: A normalized list of vectors, one for each word
        @return dict: A dictionary with a word as a key and the id for the list as the value
        """
        with open("./data/vocab.txt", 'r') as f:
            #Read in the list of words
            words = [word.rstrip().split(' ')[0] for word in f.readlines()]

        with open("./data/vectors.txt", 'r') as f:
            #word --> [vector]
            vectors = {}
            for line in f:
                vals = line.rstrip().split(' ')
                word = vals[0]
                vec = vals[1:]
                vectors[word] = [float(x) for x in vec]

        #Compute size of vocabulary
        vocab_size = len(words)
        word2id = {w: idx for idx, w in enumerate(words)}
        id2word = {idx: w for idx, w in enumerate(words)}

        vector_dim = len(vectors[id2word[0]])
        #print("Vocab size: " + str(vocab_size))
        #print("Vector dimension: " + str(vector_dim))

        #Create a numpy matrix to hold word vectors
        W = np.zeros((vocab_size, vector_dim))
        for word, v in vectors.items():
            if word == '<unk>':
                continue
            W[word2id[word], :] = v

        #Normalize each word vector to unit length
        W_norm = np.zeros(W.shape)
        d = (np.sum(W ** 2, 1) ** (0.5))
        W_norm = (W.T / d).T

        return W_norm, word2id




    def create_vect(self, observation:str) -> list:
        """
        Takes an observation and returns a 50 dimensional vector representation of it

        @param str: a string containing an observation

        @return list: A list representing a 50 dimensional normalized vector
        """
        obs_split = observation.split(' ')
        num_words = 0


        vect_size = 50
        avg_vect = [0] * vect_size

        for word in obs_split:
            if(self.word2id.get(word) is not None):
                id = self.word2id.get(word)
                norm_vect = self.vocab_vectors[id]
                avg_vect = list(map(add, avg_vect, norm_vect))
                num_words +=1
            else:
                print("Word not in the vocab: " + word)

        avg_vect = list(map(truediv, avg_vect, num_words))

        return(avg_vect)



    def create_observation_vect(self, env:FrotzEnv) -> list:
        """
        Takes the gamestate and returns a vector representing the previous observation

        @param env: the current gamestate
        @return list: A normalized vector representing the previous observation
        """
        currState = env.get_state()
        gameState = currState[8].decode()

        onlyTxt = re.sub('\n', ' ', gameState)
        onlyTxt = re.sub('[,?!.:;\'\"]', '', onlyTxt)
        onlyTxt = re.sub('\s+', ' ', onlyTxt)
        onlyTxt = onlyTxt.lower()
        #print(onlyTxt)
        onlyTxt = onlyTxt[:len(onlyTxt)-1]
        observation = onlyTxt
        avg_vect = self.create_vect(observation)

        return(avg_vect)


    def sort_actions(self, valid_actions:list) -> list:
        """
            looks through all the valid actions and sorted them by hoarder,
            mover, fighter, or everything else.

            @param valid_actions

            @return sorted_actions, list of lists of sorted actions
        """
        mover_actions = []
        fighter_actions = []
        ee_actions = []
        hoarder_actions = []
        # for each action
        for action in valid_actions:
            # check if action aligns with movements
            if action in self.movements:
               mover_actions.append(action)
            # check if action contains any enemies or weapons
            elif action in self.enemies or action in self.weapons:
                fighter_actions.append(action)
            # check if action contains "take"
            elif "take" in action:
                hoarder_actions.append(action)
            # else add to everything else action list
            else:
                ee_actions.append(action)
        # create list of lists
        sorted_actions=[hoarder_actions,
                        mover_actions,
                        fighter_actions,
                        ee_actions]
        return sorted_actions

    def get_action(self, env:FrotzEnv, valid_actions:list, history:list) -> str:
        """
        Gets the closest possible action to the observation

        @param FrotzEnv: the game environment
        @param list: the list of all valid actions
        @param list: the game history

        @return str: Returns the action closest to the observation
        """
        # get list of past actions
        past_actions = []
        if len(history) > self.PAST_ACTIONS_CHECK:
            for combo in history[self.PAST_ACTIONS_CHECK]:
                past_actions.append(combo[1])
        observation = env.get_state()[8]

        # Encode the observation
        query_vec = self.model.encode([observation])[0]

        # set up for testing all actions
        chosen_action = ""
        best_similarity = 0
        # for each vaild action, test its similarity to the observation
        for action in valid_actions:
            if (len(history) !=0 and action in past_actions):
                continue
            sim = dot(query_vec, self.model.encode([action])[0]) \
                      /(norm(query_vec)*norm(self.model.encode([action])[0]))
            # choose action with the best similarity
            if (best_similarity < sim):
                best_similarity = sim
                chosen_action = action
        return chosen_action

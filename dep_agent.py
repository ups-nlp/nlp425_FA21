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
import random


# Installed modules
from jericho import FrotzEnv
from numpy import dot
from numpy.linalg import norm
#from sentence_transformers import SentenceTransformer
import tensorflow as tf
from operator import add
from operator import truediv
from tensorflow.keras import models
import pickle
import pandas as pd

#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense
#from tensorflow.keras.utils import to_categorical
#from tensorflow.keras.optimizers import Adam
#from sklearn.model_selection import train_test_split
#from matplotlib import pyplot as plt
#import os
#import torch
#from torch import nn
#from torch.utils.data import DataLoader
#from torchvision import datasets, transforms

# In-house modules
from agent import Agent
from pca_encoder import PCAencoder
from word_transformer import glove


class DEPagent(Agent):
    """Agent created by Danielle, Eric, and Penny."""

    def __init__(self):
        """
        Initialize the class by setting instance variables for enemies,
        weapons, and movements
        """

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
                          'northeast':'southwest', 'southwest':'southeast',
                          'go':'go'}

        # index of the current observation in the enviroment
        self.OBSERVATION_INDEX = 8

        # Number of past action to check
        self.PAST_ACTIONS_CHECK = 3

        self.vocab_vectors, self.word2id = embed_vocab()

        #Find the NN files that will save weights to a file
        self.reconstructed_model = tf.keras.models.load_model('./NN/dm_nn')

        # set model for sentence transformers
        #self.model = SentenceTransformer('all-MiniLM-L6-v2')

        # Load in the EE model (where should this happen?)
        self.ee_model = models.load_model('NN/ee_neural_network_model')

        # We need the unique actions for the ee neural network
        with open('NN/unique_actions.pkl', 'rb') as fid:
            self.unique_actions = pickle.load(fid)

        # The training file to use to create the autoencoder for the
        # Everything Else module. Hard-wiring for now
        INPUT_FILE = 'data/frotz_builtin_walkthrough.csv'
        df = pd.read_csv(INPUT_FILE)
        self.observations = (df['Observation']).values    # The inputs
        self.pcaEncoder = PCAencoder(self.observations)


    def hoarder(self, env:FrotzEnv, valid_actions:list, history:list) -> str:
        """
        Determine what action the hoarder would take.

        @param valid_actions
        @param history

        For the moment, hoarder will return "take all" until futher
        intelligence can be added.

        @return chosen_action: A string of containing "take all"

        """
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
        return self.get_action(env, valid_actions, history)

    def mover(self, env:FrotzEnv, valid_actions:list, history:list) -> str:
        """
        @param valid_actions
        @param history

        The mover will take some set of valid move actions, cross that with
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

        return self.get_action(env, valid_movements, history)


    def everything_else(self, env: FrotzEnv, valid_actions:list, \
                        history:list) -> str:
        """
        Feed the observation and list of vaild actions into a model
        (similarity / neural network) that will decide what the next
        action will be.
        @param env The enviroment
        @param valid_actions
        @param history
        @return chosen_action: A String containing a new action
        """

        # Assign probabilities to all the valid actions, make them equally
        # probable

        prob = [1/len(valid_actions) for i in range(len(valid_actions))]

        # The current observation is not in the history yet! Get from here:
        curr_obs = str(env.get_state()[-1])

        # Autoencode the observation
        encoded_obs = self.pcaEncoder.encode(np.array([curr_obs]))

        # Run the neural network to choose an action
        # Give these a much lower probability
        predict = 0.1 * self.ee_model.predict(encoded_obs)

        # Add these probabilities to the list
        prob += predict[0]

        # Normalize
        prob = prob/np.sum(prob)

        # Roll the dice
        prob_sum = np.zeros(len(prob))
        prob_sum[0] = prob[0]
        randnum = random.random()
        for i in range(1,len(prob)):
            prob_sum[i] = prob_sum[i-1] + prob[i]
            if randnum >= prob_sum[i-1] and randnum <= prob_sum[i]:
                break

        if i < len(valid_actions):
            chosen_action = valid_actions[i]
        else:
            chosen_action = self.unique_actions[i-len(valid_actions)]


        return chosen_action


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

        chosen_module = self.decision_maker(valid_actions, env, history)

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
             chosen_module = 1

        action = action_modules[chosen_module](env,
                                               sorted_actions[chosen_module],
                                               history)
        return action


    def decision_maker(self, valid_actions:list, env: FrotzEnv, history:list) -> int:
        """
        Decide which choice to take.

        @param valid_actions
        @param environment

        Creates an embedding of all the words in the previous observation,
        runs that through a neural network that ranks how much we should use
        each of the modules. Then returns an int that represents the module
        with the highest value that has valid actions

        @return chosen_module: an integer between 0 and 3, representing the
                               action module we are going to use
        """

        vector = self.create_observation_vect(env)
        np_vector = np.array([vector])

        sorted_actions = self.sort_actions(valid_actions)


        prediction = self.reconstructed_model.predict(np_vector)

        #0 at the end because its a 2D array for some reason
        sorted_prediction = np.ndarray.argsort(prediction)[0]
        reverse_sorted_prediction = sorted_prediction[::-1]
        print(prediction)
        print(sorted_prediction)
        print(reverse_sorted_prediction)

        hist_len = len(history)

        #0 is hoarder
        #1 is mover
        #2 is fighter
        #3 is everything else
        module_rank_num = 0
        while(module_rank_num < 4):
            module_num = reverse_sorted_prediction[module_rank_num]
            rand_val = random.random()
            if (rand_val < .25) and module_num == 3:
                print("hit rand val")
                module_rank_num+=1
                continue

            #if there are actions for that module or is the everything else module
            num_actions = len(sorted_actions[module_num])
            #is_ee_module = module_num == 3
            if(num_actions > 0):
                if hist_len > 2:
                    if history[hist_len-1] == history[hist_len-2]:
                        print(reverse_sorted_prediction[0])
                        rand_int = random.randint(0,3)
                        print("Chose random module: " + str(rand_int))
                        return rand_int
                    elif history[hist_len-1] == history[hist_len-3]:
                        print(reverse_sorted_prediction[0])
                        rand_int = random.randint(0,3)
                        print("Chose random module: " + str(rand_int))
                        return rand_int

                print("Top choice: " + str(reverse_sorted_prediction[0]))
                print("Chosen module: " + str(module_num))
                return module_num

            module_rank_num+=1


        #chosen_module = random.randint(0, 3)
        #return chosen_module


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

        #Remove the newline character
        onlyTxt = onlyTxt[:len(onlyTxt)-1]
        observation = onlyTxt

        #Call the create_vect method to turn the string into a list representing a vector
        #avg_vect = create_vect(self.vocab_vectors, self.word2id, observation)
        avg_vect = glove.create_vect(self, observation)

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
        # get list of select number of past actions
        # to prevent repeated the same action within that spread
        past_actions = []
        if len(history) > self.PAST_ACTIONS_CHECK:
            for combo in history[self.PAST_ACTIONS_CHECK]:
                past_actions.append(combo[1])

        # get the oberservation
        currState = env.get_state()
        gameState = currState[8].decode()
        observation = glove.create_vect(self, gameState)
        #glove.create_vect(env)
        #observation = self.creat_vect

        # Encode the observation
        #query_vec = self.model.encode([observation])[0]

        # set up for testing all actions
        chosen_action = ""
        best_similarity = 0
        # for each vaild action, test its similarity to the observation
        for action in valid_actions:
            if (len(history) !=0 and action in past_actions):
                continue
            action_vec = glove.create_vect(self, action)
            sim = dot(observation, action_vec) \
                      /(norm(observation)*norm(action_vec))
            # choose action with the best similarity
            if (best_similarity < sim):
                best_similarity = sim
                chosen_action = action
        # return the action with the closest similarity
        return chosen_action


def embed_vocab() -> (list, dict):
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


def create_vect(vocab_vectors:list, word2id:dict, observation:str) -> list:
    """
    Takes an observation and returns a 50 dimensional vector representation of it

    @param list: a list containing the vectors in the vocab which are represented by lists
    @param dict: a dictionary linking a word to a vector
    @param str: a string containing an observation

    @return list: A list representing a 50 dimensional normalized vector
    """
    obs_split = observation.split(' ')
    num_words = 0

    #Creates an empty list of size 50 to be filled in
    vect_size = 50
    avg_vect = [0] * vect_size

    for word in obs_split:
        #Check if the word is in our vocab, if it is add it to the vector
        if(word2id.get(word) is not None):
            id = word2id.get(word)
            norm_vect = vocab_vectors[id]
            avg_vect = list(map(add, avg_vect, norm_vect))
            num_words +=1
        else:
            print("Word not in the vocab: " + word)

    words = [num_words] * vect_size
    avg_vect = list(map(truediv, avg_vect, words))

    return(avg_vect)

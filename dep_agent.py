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
                          'northeast':'southwest', 'southwest':'southeast'}


        self.vocab_vectors, self.word2id = self.embed_vocab()

        #NEURAL NET STUFF HERE




        # Train model
        # The history is a list of (observation, action) tuples
        #model= []
        #curr_obs, info = env.reset()
        #done = False
        #env.reset(True)
        #winning_actions = len(env.get_walkthrough())
        #for action in winning_actions:
            # For each step of game play, the agent determines the next action
            # based on env and the history of observations and actions
            # env is the environment from Frotz
         #   action_to_take = action
            # info is a dictionary (i.e. hashmap) of {'moves':int, 'score':int}
         #   next_obs, _, done, info = env.step(action_to_take)

         #   history.append((curr_obs, action_to_take))

         #   curr_obs = next_obs

        self.model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')
        #env.reset()


    def hoarder(self, env:FrotzEnv, valid_actions:list, history:list) -> str:
        """
        Determine what action the hoarder would take.

        @param valid_actions
        @param history

        For the moment, hoarder will return "take all" until futher
        intelligence can be added.

        @return chosen_action: A string of containing "take all"

        """
        return 'take all'


    def fighter(self, env:FrotzEnv, valid_actions:list, history:list) -> str:
        """
        @param env: The environment variable, which contains the inventory
        @param valid_actions: The list of valid actions
        @param history: A list of tuples (observation, action)

        For the moment: fighter will return "kill ___ with ____" if an enemy
        and a weapon are identified. Otherwise it will run.

        @return chosen_action: String containing "kill ____ with ____" or "run"
        """

        inventory = [item.name for item in env.get_inventory()]
        for action in valid_actions:
            have_enemy = [enemy in action for enemy in self.enemies]
            if any(have_enemy):
                enemy = self.enemies[have_enemy.index(True)]
                # We are facing an enemy: if we have a weaon, fight with it
                # if not, run
                have_weapons = [weapon in inventory for weapon in self.weapons]
                if any(have_weapons):
                    weapon = self.weapons[have_weapons.index(True)]
                    return 'kill ' + enemy + ' with ' + weapon
                return 'run'
        return 'run'


    def mover(self, env:FrotzEnv, valid_actions:list, history:list) -> str:
        """
        @param valid_actions
        @param history

        The mover will take some set of vaild move actions, cross that will
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
        return random.choice(valid_movements)


    def everything_else(self, env: FrotzEnv, valid_actions:list, \
                        history:list) -> str:
        """
        Feed the observation and list of vaild actions into a recurrent
        neural network that will decided what the next action will be

        @param valid_actions
        @param history

        @return chosen_action: A String containing a new action
        """
        # get list of past actions
        if len(history) != 0:
            past_actions = []
            for combo in history:
                past_actions.append(combo[1])
        observation = env.get_state()[8]
        query_vec = self.model.encode([observation])[0]
        # set up for testing all actions
        chosen_action = ""
        best_similarity = 0
        # for each vaild action, tests its similarity to the observation
        for action in valid_actions:
            sim = dot(query_vec, self.model.encode([action])[0])/(norm(query_vec)*norm(self.model.encode([action])[0]))
            # chooese action with the best similarity
            if(best_similarity < sim):
                if(len(history) !=0 and past_actions.count(action)>1):
                    continue
                best_similarity = sim
                chosen_action = action
        print(" similarity: ", best_similarity)
        return chosen_action    # action with the best similarity to the observation



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

        chosen_module = self.decision_maker(valid_actions, history, env)
        #chosen_module = 3

        action_modules = [self.hoarder,
                          self.mover,
                          self.fighter,
                          self.everything_else]
        action = action_modules[chosen_module](env, valid_actions, history)

        return action


    def decision_maker(self, valid_actions:list, history:list, env: FrotzEnv) -> int:
        """
        Decide which choice to take randomly for now
        this needs some intelligence

        @param valid_actions
        @param history

        Creates an embedding of all the words in the previous observation,
        runs that through a neural network that ranks how much we should use
        each of the modules. Then returns an int that represents the module
        with the highest value that has valid actions

        @return chosen_module: an integer between 0 and 3, representing the
                               action module we are going to use
        """

        vector = self.create_observation_vect(env)


        chosen_module = random.randint(0, 3)
        return chosen_module


    def embed_vocab(self) -> (list, dict):
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


    def create_observation_vect(self, env:FrotzEnv) -> list:
        currState = env.get_state()
        gameState = currState[8].decode()
        if gameState.startswith("Copyright"):
            index = gameState.index("West")
            gameState = gameState[index:]
        onlyTxt = re.sub('\n', ' ', gameState)
        onlyTxt = re.sub('[,?!.:;\'\"]', '', onlyTxt)
        onlyTxt = re.sub('\s+', ' ', onlyTxt)
        onlyTxt = onlyTxt.lower()
        #print(onlyTxt)
        onlyTxt = onlyTxt[:len(onlyTxt)-1]
        observation = onlyTxt

        obs_split = observation.split(' ')
        avg_vect = []
        vect_size = 50

        i=0
        while i < vect_size:
            avg_vect.append(0)
            i+=1

        for word in obs_split:
            if(self.word2id.get(word) is not None):
                id = self.word2id.get(word)
                norm_Vect = self.vocab_vectors[id]

                i=0
                for vect in norm_Vect:
                    curr = avg_vect[i]
                    val = vect + curr
                    avg_vect[i] = val
                    i+=1
            else:
                print("Word not in the vocab: " + word)

        totalVal = 0
        for val in avg_vect:
            totalVal += val

        i=0
        while i<vect_size:
            val = avg_vect[i]
            avg_val = (val/totalVal)
            avg_vect[i] = avg_val
            i+=1

        return(avg_vect)

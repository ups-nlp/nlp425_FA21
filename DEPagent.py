#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 18:59:14 2021

@author: prowe
"""

# Built-in modules

# Installed modules
import random
from jericho import FrotzEnv
#import spacy
#import benepar; benepar.download('benepar_en3')

# # Set up nlp
# nlp = spacy.load('en_core_web_lg')
# if spacy.__version__.startswith('2'):
#     nlp.add_pipe(benepar.BeneparComponent("benepar_en3"))
# else:
#     nlp.add_pipe("benepar", config={"model": "benepar_en3"})
    

# In-house modules
from agent import Agent


class DEPagent(Agent):
    """Agent created by Danielle, Eric, and Penny. Details TBD"""
    
    def hoarder(self, valid_actions, history):
        """ 
        Determine what action the hoarder would take.
        
        Questions:
        1) Should this be a method or a class?
        2) Don't we need the points as input, so we can train?
        """
        return 'take all'
    
    def interactor(self, valid_actions, history):
        """ Determine what action the interactor would take. """
        return 'attack'
    
    def observer(self, valid_actions, history):
        """ Determine what action the interactor would take. """
        return 'go  ' + random.choice(['north','south','east','west'])
    
    def decision_maker(self, actions, history):
        """ Decide which choice to take randomly for now
        this needs some intelligence
        """
        return random.choice(actions)
        
    
    def take_action(self, env: FrotzEnv, history: list) -> str:
        """Takes in the history and returns the next action to take"""

        # Eventually the valid actions will be determined by Team 1,
        # but for now use the FrotzEnv
        valid_actions = env.get_valid_actions()
        
        #for sentence in valid_actions:
        #    print(nlp(sentence))
        
        # Run a logic module that will do certain tasks every time
        # Should we parse the list of actions and get out nouns?
        
        
        # Make the possible set of actions a list of strings, so we can grow 
        # it later if we want
        actions = []
        
        # Possible responses to action from the Hoarder
        actions.append(self.hoarder(valid_actions, history))
        
        # Possible responses to action from the Observer
        actions.append(self.observer(valid_actions, history))
        
        # Possible responses to action from the Interactor
        actions.append(self.interactor(valid_actions, history))
        
        # Choose between the hoarder, observer, and interactor
        return self.decision_maker(actions, history)

    
    



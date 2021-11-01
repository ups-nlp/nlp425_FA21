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

    
    



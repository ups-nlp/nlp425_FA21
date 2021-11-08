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
from decision_maker import decision_maker



class DEPagent(Agent):
    """Agent created by Danielle, Eric, and Penny. Details TBD"""

    def hoarder(self, valid_actions:list, history:list) -> str:
        """
        Determine what action the hoarder would take.

        @param valid_actions
        @param history

        For the moment, hoarder will return "take all" until futher intelligance
        can be added.

        @return chosen_action: A string of containing "take all"

        """
        return 'take all'

    def fighter(self, valid_actions:list, history:list) -> str:
        """
        @param valid_actions
        @param history

        For the moment: fighter will return "kill ___ with ____" until further
        intelligance can be added.

        @return chosen_action: A string containing "kill ____ with ____"
        """
        return 'kill ___ with ___'

    def mover(self, valid_actions:list, history:list) -> str:
        """
        @param valid_actions
        @param history

        The mover will take some set of vaild move actions, cross that will
        other known move actions and will pick one by prioritizing new directions

        @return chosen_action: A string containing a move action
        """
        return 'go  ' + random.choice(['north','south','east','west'])


    def everything_else(self, valid_actions:list, history:list) -> str:
        """
        Feed the observation and list of vaild actions into a recurrent
        neural network that will decided what the next action will be

        @param valid_actions
        @param history

        @return chosen_action: A String containing a new action
        """
        return 'talk'



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



        chosen_module = decision_maker(valid_actions, history)

        action_modules = [self.hoarder,
                          self.mover,
                          self.fighter,
                          self.everything_else]

        action = action_modules[chosen_module](valid_actions, history)

        return action


    def decision_maker(valid_actions:list, history:list) -> int:
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
        chosen_module = random.randint(0, 3)
        return chosen_module

"""
Agents for playing text-based games
"""

import random
from jericho import FrotzEnv


class Agent:
    """Interface for an Agent"""

    def take_action(self, env: FrotzEnv, history: list) -> str:
        """Takes in the history and returns the next action to take"""
        raise NotImplementedError


class RandomAgent(Agent):
    """Agent randomly selects an action from list of valid actions"""

    def take_action(self, env: FrotzEnv, history: list) -> str:
        """Takes in the history and returns the next action to take"""
        valid_actions = env.get_valid_actions()
        return random.choice(valid_actions)


class HumanAgent(Agent):
    """Allows a human player"""

    def take_action(self, env: FrotzEnv, history: list) -> str:
        """Takes in the history and returns the next action to take"""
        print("Action: ")
        return input()


class DEPagent(Agent):
    """Agent created by Danielle, Eric, and Penny. Details TBD"""
    
    def hoarder(self, valid_actions, history):
        """ 
        Determine what action the hoarder would take.
        
        Questions:
        1) Should this be a method or a class?
        2) Don't we need the points as input, so we can train?
        """
        return "Hoarder's choice of actions"
    
    def decision_maker(self, actions, history):
        """ Decide which choice to take randomly for now"""
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
        actions.append("Add method call for observer's choice")
        
        # Possible responses to action from the Interactor
        actions.append("Add method call for interactor's choice")
        
        # Choose between the hoarder, observer, and interactor
        return self.decision_maker(actions, history)

    
    

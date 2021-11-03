"""
Agents for playing text-based games
"""

import random
import mcts_agent
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

class MonteAgent(Agent):
    """"Monte Carlo Search Tree Player"""


    def __init__(self, env: FrotzEnv, num_steps: int):
        # create root node with the initial state
        self.root = mcts_agent.Node(None, None, 0)
        # create a pointer node to use to traverse the tree later
        self.current = self.root
        # This constant balances tree exploration with exploitation of ideal nodes
        self.exploreConst = 2

        # train the tree using the Monte Carlo Search Algorithm
        count = 0
        # for now, we will only generate a tree with 101 nodes
        while(count < 100):
            # Create a new node on the tree
            newNode = mcts_agent.treePolicy(self.root, env, self.exploreConst)
            # Determine the simulated value of the new node
            delta = mcts_agent.defaultPolicy(newNode, env, num_steps)
            # Propogate the simulated value back up the tree
            mcts_agent.backUp(newNode, delta)

            # reset the state of the game when done with one simulation 
            env.reset()
            count += 1

    
    def take_action(self, env: FrotzEnv, history: list) -> str:
        """Takes in the history and returns the next action to take"""
        print("Action: ")
        self.current = mcts_agent.bestChild(self.current, self.exploreConst)
        return self.current.getPrevAction()


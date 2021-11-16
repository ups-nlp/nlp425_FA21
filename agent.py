"""
Agents for playing text-based games
"""

import random
from jericho import FrotzEnv
from math import sqrt
import mcts_agent


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
        print(env.get_valid_actions())
        print("Action: ")
        return input()


class MonteAgent(Agent):
    """"Monte Carlo Search Tree Player"""

    def __init__(self, env: FrotzEnv, num_steps: int):
        # create root node with the initial state
        self.root = mcts_agent.Node(None, None, env.get_valid_actions())

        # This constant balances tree exploration with exploitation of ideal nodes
        self.explore_const = 1.0/sqrt(2)

        self.reward = mcts_agent.AdditiveReward()



    def take_action(self, env: FrotzEnv, history: list) -> str:
        """Takes in the history and returns the next action to take"""
        print("Action: ")
        #
        # Train the agent using the Monte Carlo Search Algorithm
        #

        # The length of each monte carlo simulation
        simulation_length = 20

        # Maximum number of nodes to generate in the tree each time a move is made
        max_nodes = 40

        #current number of generated nodes
        count = 0
        
        #current state of the game. Return to this state each time generating a new node
        curr_state = env.get_state()
        while(count <= max_nodes):
            if(count % 10 == 0): 
                print(count)
            # Create a new node on the tree
            new_node = mcts_agent.tree_policy(self.root, env, self.explore_const)
            # Determine the simulated value of the new node
            delta = mcts_agent.default_policy(new_node, env, simulation_length, self.reward)
            # Propogate the simulated value back up the tree
            mcts_agent.backup(new_node, delta)
            # reset the state of the game when done with one simulation
            env.reset()
            env.set_state(curr_state)
            count += 1


        print(env.get_valid_actions())
        for child in self.root.children:
            print(child.get_prev_action(), ", count:", child.visited, ", value:", child.sim_value, "normalized value:", (child.sim_value/child.visited))
        self.root = mcts_agent.best_child(self.root, self.explore_const, False)
        return self.root.get_prev_action()

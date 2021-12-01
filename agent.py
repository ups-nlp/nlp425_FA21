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

    node_path = []

    def __init__(self, env: FrotzEnv, num_steps: int):
        # create root node with the initial state
        self.root = mcts_agent.Node(None, None, env.get_valid_actions())

        self.node_path.append(self.root)

        # This constant balances tree exploration with exploitation of ideal nodes
        self.explore_const = 1.0/sqrt(2)

        # The length of each monte carlo simulation
        self.simulation_length = 40

        # Maximum number of nodes to generate in the tree each time a move is made
        self.max_nodes = 40

        self.reward = mcts_agent.Dynamic_Reward()



    def take_action(self, env: FrotzEnv, history: list) -> str:
        """Takes in the history and returns the next action to take"""
        print("Action: ")
        #
        # Train the agent using the Monte Carlo Search Algorithm
        #

        #current number of generated nodes
        count = 0
        
        #current state of the game. Return to this state each time generating a new node
        curr_state = env.get_state()
        while(count <= self.max_nodes):
            if(count % 100 == 0): 
                print(count)
            # Create a new node on the tree
            new_node = mcts_agent.tree_policy(self.root, env, self.explore_const, self.reward)
            # Determine the simulated value of the new node
            delta = mcts_agent.default_policy(new_node, env, self.simulation_length, self.reward)
            # Propogate the simulated value back up the tree
            mcts_agent.backup(new_node, delta)
            # reset the state of the game when done with one simulation
            env.reset()
            env.set_state(curr_state)
            count += 1


        print(env.get_valid_actions())
        for child in self.root.children:
            print(child.get_prev_action(), ", count:", child.visited, ", value:", child.sim_value, "normalized value:", self.reward.select_action(env, child.sim_value, child.visited, None))

        ## Pick the next action
        self.root, score_dif = mcts_agent.best_child(self.root, self.explore_const, env, self.reward, False)

        self.node_path.append(self.root)

        ## Dynamically adjust simulation length based on how sure we are 
        self.max_nodes, self.simulation_length = self.reward.dynamic_sim_len(self.max_nodes, self.simulation_length, score_dif)

        print("\n\n------------------ ", score_dif, self.max_nodes, self.simulation_length)

        return self.root.get_prev_action()
"""
An implementation of the UCT algorithm for text-based games
"""

from math import inf, sqrt, log2, floor,e
import random
import sys
import numpy as np
from jericho import FrotzEnv

def tree_policy(root, env: FrotzEnv, explore_exploit_const, reward_policy):
    """ Travel down the tree to the ideal node to expand on

    This function loops down the tree until it finds a
    node whose children have not been fully explored, or it
    explores the best child node of the current node.

    Keyword arguments:
    root -- the root node of the tree
    env -- FrotzEnv interface between the learning agent and the game
    Return: the ideal node to expand on
    """
    node = root
    # How do you go back up the tree to explore other paths
    # when the best path has progressed past the max_depth?
    #while env.get_moves() < max_depth:
    while not node.is_terminal():
        #if parent is not full expanded, expand it and return
        if not node.is_expanded():
            return expand_node(node, env)
        #Otherwise, look at the parent's best child
        else:
            # Select the best child of the current node to explore
            child = best_child(node, explore_exploit_const, env, reward_policy)[0]
            # else, go into the best child
            node = child
            # update the env variable
            env.step(node.get_prev_action())

    # The node is terminal, so return it
    return node

def best_child(parent, exploration, env: FrotzEnv, reward_policy, use_bound = True):
    """ Select and return the best child of the parent node to explore or the action to take

    From the current parent node, we will select the best child node to
    explore and return it. The exploration constant is inputted into this function,
    it balances exploration with exploitation. If the parent node has unexplored
    children, they will automatically be explored first.

    or 

    From the availble actions from this node, we will pick the one that has the most 
    efficient score / visited ratio. Aka the best action to take

    Keyword arguments:
    parent -- the parent node
    exploration -- the exploration-exploitation constant
    use_bound -- whether you are picking the best child to expand (true) or selecting the best action (false)
    Return: the best child to explore in an array with the difference in score between the first and second pick
    """
    max_val = -inf
    bestLs = [None]
    second_best_score = -inf
    for child in parent.get_children():
        # Use the Upper Confidence Bounds for Trees to determine the value for the child or pick the child based on visited
        if(use_bound):
            child_value = reward_policy.upper_confidence_bounds(env, exploration, child.sim_value, child.visited, parent.visited)
        else:
            child_value = reward_policy.select_action(env, child.sim_value, child.visited, parent.visited)
        
        #print("child_value", child_value)
        # if there is a tie for best child, randomly pick one
        # if(child_value == max_val) with floats
        if (abs(child_value - max_val) < 0.000000001):
            
            #print("reoccuring best", child_value)
            #print("next best", child_value)
            bestLs.append(child)
            second_best_score = child_value
            
        #if it's value is greater than the best so far, it will be our best so far
        elif child_value > max_val:
            #print("new best", child_value)
            #print("next best", max_val)
            second_best_score = max_val
            bestLs = [child]
            max_val = child_value
        #if it's value is greater than the 2nd best, update our 2nd best
        elif child_value > second_best_score:
            #print("best", bestLs[0])
            #print("new next best", child_value)
            #print("old next best", second_best_score)
            second_best_score = child_value
    chosen = random.choice(bestLs)
    if( not use_bound):
        print("best, second", max_val, second_best_score)
    return chosen, abs(max_val - second_best_score) ## Worry about if only 1 node possible infinity?


def expand_node(parent, env):
    """
    Expand this node

    Create a random child of this node

    Keyword arguments:
    parent -- the node being expanded
    env -- FrotzEnv interface between the learning agent and the game
    Return: a child node to explore
    """
    # Get possible unexplored actions
    actions = parent.new_actions 

    #print(len(actions), rand_index)
    action = random.choice(actions)

    # Remove that action from the unexplored action list and update parent
    actions.remove(action)

    # Step into the state of that child and get its possible actions
    env.step(action)
    new_actions = env.get_valid_actions()

    # Create the child
    new_node = Node(parent, action, new_actions)

    # Add the child to the parent
    parent.add_child(new_node)

    return new_node


    # # if no new nodes were created, we are at a terminal state
    # if new_node is None:
    #     # set the parent to terminal and return the parent
    #     parent.terminal = True
    #     return parent

    # else:
    #     # update the env variable to the new node we are exploring
    #     env.step(new_node.get_prev_action())
    #     # Return a newly created node to-be-explored
    #     return new_node


def default_policy(new_node, env, sim_length, reward_policy):
    """
    The default_policy represents a simulated exploration of the tree from
    the passed-in node to a terminal state.

    Self-note: This method doesn't require the nodes to store their depth
    """
    #if node is already terminal, return 0
    if(env.game_over()):
        return reward_policy.terminal_node(env)
    # While the game is not over and we have not run out of moves, keep exploring
    while (not env.game_over()) and (not env.victory()):

        # if we have reached the limit for exploration
        if(env.get_moves() < sim_length):
            #return the reward received by reaching terminal state
            return reward_policy.simulation_limit(env)

        #INIT. DEFAULT POLICY: explore a random action from the list of available actions.
        #Once an action is explored, remove from the available actions list
        # Select a random action from this state
        actions = env.get_valid_actions()
        # Take that action, updating env to a new state
        env.step(random.choice(actions))

    #return the reward received by reaching terminal state 
    # (add 10 to score to counteract the -10 punishment for dying)
    return reward_policy.simulation_terminal(env)


def backup(node, delta):
    """
    This function backpropogates the results of the Monte Carlo Simulation back up the tree

    Keyword arguments:
    node -- the child node we simulated from
    delta -- the component of the reward vector associated with the current player at node v
    """
    while node is not None:
        # Increment the number of times the node has
        # been visited and the simulated value of the node
        node.visited += 1
        node.sim_value += delta
        # Traverse up the tree
        node = node.get_parent()

class Node:
    """
    This Node class represents a state of the game. Each node holds the following:
    parent -- it's parent node
    prev_act -- the previous action taken to get to this node
    children -- a list of the children of this node
    sim_value -- the simulated value of the node
    visited -- the number of times this node has been visited
    terminal -- a boolean indicating if this node is terminal

    Keyword arguments:
    parent -- it's parent node
    prev_act -- the previous action taken to get to this node
    """

    def __init__(self, parent, prev_act, new_actions):
        self.parent = parent
        self.prev_act = prev_act
        self.children = []
        self.sim_value = 0
        self.visited = 0
        self.max_children = len(new_actions)
        self.new_actions = new_actions

    # The node is terminal if it has no children and no possible children
    def is_terminal(self):
        return self.max_children == 0

    def print(self, level):
        space = ">" * level
        #for i in range(level):
        #    space += ">"
        if self.prev_act is None:
            print("\t"+space+"<root>"+"\n")
        else:
            print("\t"+space+self.prev_act+"\n")

        for child in self.children:
            child.print(level+1)

    def add_child(self, child):
        self.children.append(child)

    def get_parent(self):
        return self.parent

    def get_prev_action(self):
        return self.prev_act

    def get_children(self):
        return self.children

    # Return true if it has expanded all possible actions AND has at least 1 child
    def is_expanded(self):
        #print("is expanded: ", len(self.new_actions), len(self.children))
        return (len(self.children) == self.max_children)

class Reward:
    """Interface for a Reward"""

    def terminal_node(self, env) -> int:
        """ The case when we start the simulation at a terminal state """
        raise NotImplementedError

    def simulation_limit(self, env) -> int:
        """ The case when we reach the simulation depth limit """
        raise NotImplementedError

    def simulation_terminal(self, env) -> int:
        """ The case when we reach a terminal stae in the simulation """
        raise NotImplementedError

    def dynamic_sim_len(self, max_nodes, sim_limit, diff) -> int:
        """ Given the current simulation depth limit and the difference between the picked and almost picked 'next action' return what the new sim depth is """
        raise NotImplementedError
        
    def upper_confidence_bounds(self, env: FrotzEnv, exploration, child_sim_value, child_visited, parent_visited) -> int:
        raise NotImplementedError

    def select_action(self, env: FrotzEnv, child_sim_value, child_visited, parent_visited) -> int:
        raise NotImplementedError

class Softmax_Reward:
    """Softmax reward returns values from 0 to .5 for the state. 
    This implementation assumes that every score between the loss state and the max score
    are possible.
    """

    def terminal_node(self, env):
        """ The case when we start the simulation at a terminal state """
        return 0

    def simulation_limit(self, env):
        """ The case when we reach the simulation depth limit """
        return env.get_score()

    def simulation_terminal(self, env):
        """ The case when we reach a terminal state in the simulation """
        raise (env.get_score()+10)

    def dynamic_sim_len(self, max_nodes, sim_limit, diff):
        """ Given the current simulation depth limit and the difference between the picked and almost picked 'next action' return what the new sim depth is """
        new_limit = sim_limit
        new_node_max = max_nodes
        if(diff < 0.1):
            if(new_node_max < 1000):
                new_node_max = max_nodes*2

            
            if(new_node_max < 10000):
                new_limit = new_limit*2
            

        elif(diff > 2):
            if(new_node_max > 300):
                new_node_max = floor(max_nodes/2)
            
            if(sim_limit > 10):
                new_limit =  floor(sim_limit/2)
        
        return new_node_max, new_limit

    def softmax_calc(self,minScore,maxScore):
        total = 0
        for i in range (minScore,maxScore+1):
            total = total+(e**i)
        return total

        
    def upper_confidence_bounds(self, env: FrotzEnv, exploration, child_sim_value, child_visited, parent_visited):
        if env.get_score() >= np.log(sys.maxsize):
            denom = np.log(sys.maxsize)
        else:
            denom = self.softmax_calc(-10,env.get_max_score())
        if child_sim_value >= np.log(sys.maxsize):
            num = np.log(sys.maxsize)
        else:
            num = child_sim_value
        
        return (e**(num))/(child_visited*denom)+ exploration*sqrt((2*log2(parent_visited))/child_visited)

    def select_action(self, env: FrotzEnv, child_sim_value, child_visited, parent_visited):
        if env.get_max_score() >= np.log(sys.maxsize):
            denom = np.log(sys.maxsize)
        else:
            denom = self.softmax_calc(-10,env.get_max_score())
        if child_sim_value >= np.log(sys.maxsize):
            num = np.log(sys.maxsize)
        else:
            num = child_sim_value
        return (e**(num))/(child_visited*denom)

class Generalized_Softmax_Reward:
    """Generalized Softmax reward returns values from 0 to 1 for the state. 
    This implementation assumes that every score between the loss state and the max score
    are possible.
    """

    def terminal_node(self, env):
        """ The case when we start the simulation at a terminal state """
        return 0

    def simulation_limit(self, env):
        """ The case when we reach the simulation depth limit """
        return env.get_score()

    def simulation_terminal(self, env):
        """ The case when we reach a terminal state in the simulation """
        raise (env.get_score()+10)

    def dynamic_sim_len(self, max_nodes, sim_limit, diff):
        """ Given the current simulation depth limit and the difference between the picked and almost picked 'next action' return what the new sim depth is """
        new_limit = sim_limit
        new_node_max = max_nodes
        if(diff < 0.1):
            if(new_node_max < 1000):
                new_node_max = max_nodes*2

            
            if(new_node_max < 10000):
                new_limit = new_limit*2
            

        elif(diff > 2):
            if(new_node_max > 300):
                new_node_max = floor(max_nodes/2)
            
            if(sim_limit > 10):
                new_limit =  floor(sim_limit/2)
        
        return new_node_max, new_limit
    
    def upper_confidence_bounds(self, env: FrotzEnv, exploration, child_sim_value, child_visited, parent_visited):
        if env.get_score() >= np.log(sys.maxsize):
            denom = np.log(sys.maxsize)
        else:
            denom = e**(env.get_score())
        if child_sim_value >= np.log(sys.maxsize):
            num = np.log(sys.maxsize)
        else:
            num = child_sim_value
        try:
            return (1/child_visited)*(e**(num-denom)) + exploration*sqrt((2*log2(parent_visited))/child_visited)
        except OverflowError:
            print("max size = ",sys.maxsize," num = ",num," denom = ",denom)

    def select_action(self, env: FrotzEnv, child_sim_value, child_visited, parent_visited):
        if env.get_score() >= np.log(sys.maxsize):
            denom = np.log(sys.maxsize)
        else:
            denom = e**(env.get_score())
        if child_sim_value >= np.log(sys.maxsize):
            num = np.log(sys.maxsize)
        else:
            num = child_sim_value
        try:
            return (1/child_visited)*(e**(num-denom))
        except OverflowError:
            print("max size = ",sys.maxsize," num = ",num," denom = ",denom)

class Additive_Reward(Reward):
    """This Reward Policy returns values between 0 and 1 
    for the state inputted state.

    Args:
        Reward: Reward Class Interface
    """
    def terminal_node(self, env):
        return 0

    def simulation_limit(self, env):
        return env.get_score()

    def simulation_terminal(self, env):
        """Add 10 to the score so it is non-negative"""
        return (env.get_score()+10)

    def dynamic_sim_len(self, max_nodes, sim_limit, diff) -> int: ## NEEDS LOGIC IMPLEMENTED
        new_limit = sim_limit
        new_node_max = max_nodes
        if(diff < 0.1):
            if(new_node_max < 1000):
                new_node_max = max_nodes*2

            
            if(new_node_max < 10000):
                new_limit = new_limit*2
            

        elif(diff > 2):
            if(new_node_max > 300):
                new_node_max = floor(max_nodes/2)
            
            if(sim_limit > 10):
                new_limit =  floor(sim_limit/2)
        
        return new_node_max, new_limit

    def upper_confidence_bounds(self, env: FrotzEnv, exploration, child_sim_value, child_visited, parent_visited):
        return child_sim_value/(child_visited*env.get_max_score()) + exploration*sqrt((2*log2(parent_visited))/child_visited)

    def select_action(self, env: FrotzEnv, child_sim_value, child_visited, parent_visited):
        return child_sim_value/(child_visited*env.get_max_score())

class Dynamic_Reward:
    """Dynamic Reward  scales the reward returned in a simulation by the length of the simulation,
        so a reward reached earlier in the game will have a higher score than the same state
         reached later."""

    def terminal_node(self, env) -> int:
        """ The case when we start the simulation at a terminal state """
        return 0

    def simulation_limit(self, env) -> int:
        """ The case when we reach the simulation depth limit """
        return (env.get_score()/(env.get_moves()+1))

    def simulation_terminal(self, env) -> int:
        """ The case when we reach a terminal stae in the simulation """
        return ((env.get_score()+10)/(env.get_moves()+1))

    def dynamic_sim_len(self, max_nodes, sim_limit, diff) -> int:
        """ Given the current simulation depth limit and the difference between the picked and almost picked 'next action' return what the new sim depth is """
        new_limit = sim_limit
        new_node_max = max_nodes
        if(diff < 0.1):
            if(new_node_max < 1000):
                new_node_max = max_nodes*2

            
            if(new_node_max < 10000):
                new_limit = new_limit*2
            

        elif(diff > 2):
            if(new_node_max > 300):
                new_node_max = floor(max_nodes/2)
            
            if(sim_limit > 10):
                new_limit =  floor(sim_limit/2)
        
        return new_node_max, new_limit
        
    def upper_confidence_bounds(self, env: FrotzEnv, exploration, child_sim_value, child_visited, parent_visited) -> int:
        return child_sim_value/(child_visited*env.get_max_score()) + exploration*sqrt((2*log2(parent_visited))/child_visited)

    def select_action(self, env: FrotzEnv, child_sim_value, child_visited, parent_visited) -> int:
       return (child_sim_value/(child_visited*env.get_max_score()))
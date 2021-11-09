"""
An implementation of the UCT algorithm for text-based games
"""

from math import inf, sqrt, log2, floor
import random
from jericho import FrotzEnv


def tree_policy(root, env: FrotzEnv, explore_exploit_const):
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
            child = best_child(node, explore_exploit_const)
            # else, go into the best child
            node = child
            # update the env variable
            env.step(node.get_prev_action())

    # The node is terminal, so expand it
    return expand_node(node, env)

def best_child(parent, exploration, use_bound = True):
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
    Return: the best child to explore
    """
    max_val = -inf
    bestLs = [None]
    for child in parent.get_children():
        # Use the Upper Confidence Bounds for Trees to determine the value for the child or pick the child based on visited
        if(use_bound):
            child_value = (child.sim_value/child.visited) + 2*exploration*sqrt((2*log2(parent.visited))/child.visited)
        else:
            child_value = (child.sim_value/child.visited) #select_action
        
        # if there is a tie for best child, randomly pick one
        if child_value == max_val:
            bestLs.append(child)
            max_val = child_value
        #if it's calue is greater than the best so far, it will be our best so far
        elif child_value > max_val:
            bestLs = [child]
            max_val = child_value
    return bestLs[random.randint(0, len(bestLs) - 1)]



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

    # Pick a random unexplored action
    rand_index = floor(len(actions)*random.random())
    #print(len(actions), rand_index)
    action = actions[rand_index]

    # Remove that action from the unexplored action list and update parent
    actions.remove(action)

    # Step into the state of that child and get its possible actions
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


def default_policy(new_node, env, sim_length):
    """
    The default_policy represents a simulated exploration of the tree from
    the passed-in node to a terminal state.

    Self-note: This method doesn't require the nodes to store their depth
    """
    #if currently unexplored node, set score to 0
    new_node.sim_value = 0

    prev_score = env.get_score()
 
    # While the game is not over and we have not run out of moves, keep exploring
    while (not env.game_over()) and (not env.victory()) and (env.get_moves() < sim_length):        
        #INIT. DEFAULT POLICY: explore a random action from the list of available actions.
        #Once an action is explored, remove from the available actions list
        
        # Select a random action from this state
        actions = env.get_valid_actions()
        index = random.randint(0, len(actions))
        act = actions[index-1]

        # Take that action, updating env to a new state
        env.step(act)
        outcome = env.get_score()
        if outcome > prev_score:
            return outcome

    #return the reward received by reaching terminal state
    return env.get_score()

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
        self.new_actions = new_actions

    # Sets the 'new_actions' which are unexplored actions. Basically future potential child nodes
    def set_new_actions(self, new_actions):
        self.new_actions = new_actions

    # The node is terminal if it has no children and no possible children
    def is_terminal(self):
        return (len(self.children) == 0) and (len(self.new_actions) == 0) 

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
        return len(self.new_actions) == 0 and len(self.children) != 0



############################### STUFF TO RUN DOWN SIMULATION BELOW

def reward(curr, terminal):
    """
    The reward method calculates the change in the score of the game from
    the current state to the end of the simulation.
    """
    return terminal.get_score()-curr

#######################

   


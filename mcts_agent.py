
from math import inf, sqrt, log2, trunc
from jericho import FrotzEnv
import random

""" Travel down the tree to the ideal node to expand on

This function loops down the tree until it finds a 
node whose children have not been fully explored, or it 
explores the best child node of the current node. 

Keyword arguments:
root -- the root node of the tree
env -- FrotzEnv interface between the learning agent and the game
Return: the ideal node to expand on """
def treePolicy(root, env, max_depth, explore_exploit_const):
    node = root
    # How do you go back up the tree to explore other paths 
    # when the best path has progressed past the max_depth?
    #while env.get_moves() < max_depth:
    while True:
        #if parent is not full expanded, expand it and return
        if(not node.isExpanded()):
            return expandNode(node, env)
        #Otherwise, look at the parent's best child
        else:
            # Select the best child of the current node to explore
            child = bestChild(node, explore_exploit_const)
            if child is None:
                # if all the child nodes are terminal, set this 
                # node to terminal and go back up to the parent
                node.terminal = True
                node = node.getParent()
            else:
                # else, go into the best child
                node = child
                # update the env variable
                env.step(node.getPrevAction())



""" Select and return the best child of the parent node to explore 

From the current parent node, we will select the best child node to 
explore and return it. The exploration constant is inputted into this function, 
it balances exploration with exploitation. If the parent node has unexplored 
children, they will automatically be explored first.

Keyword arguments:
parent -- the parent node 
exploration -- the exploration-exploitation constant
Return: the best child to explore """
def bestChild(parent, exploration):
    max = -inf
    bestChild = None
    for child in parent.getChildren():
        # if there is a child we haven't explored, explore it
        if(child.visited == 0):
            return child
        #if the child is terminal, go to the next child
        elif(child.terminal):
            continue
        # otherwise, check if this is the best child so far
        else:
            # Use the Upper Confidence Bounds for Trees to determine the value for the child
            childValue = (child.simValue/child.visited) + 2*exploration*sqrt((2*log2(parent.visited))/child.visited)
            # if there is a tie for best child, randomly pick one
            if(childValue == max and random.random() > .5):
                bestChild = child
                max = childValue
            #if it's calue is greater than the best so far, it will be our best so far
            elif(childValue > max):
                bestChild = child
                max = childValue
    return bestChild



""" 
Expand this node 

Create all children of the given node

Keyword arguments:
parent -- the node being expanded
env -- FrotzEnv interface between the learning agent and the game
Return: a child node to explore 
"""
# Expand all possible children of the given node
def expandNode(parent, env):
    # Get possible actions
    actions = env.get_valid_actions()
    
    # Create all possible child nodes
    newNode = None
    for action in actions: 
        # Make a new node
        newNode = Node(parent, action) 
        parent.addChild(newNode)

    # if no new nodes were created, we are at a terminal state
    if newNode is None:
        # set the parent to terminal and return the parent
        parent.terminal = True
        return parent

    else:
        # update the env variable to the new node we are exploring
        env.step(newNode.getPrevAction())
        # Return a newly created node to-be-explored
        return newNode




def defaultPolicy(newNode, env, simLength):
    """
    The defaultPolicy represents a simulated exploration of the tree from
    the passed-in node to a terminal state. 

    Self-note: This method doesn't require the nodes to store their depth
    """
    #if currently unexplored node, set score to 0
    newNode.simvalue = 0

    prevScore = env.get_score()
    
    # While the game is not over and we have not run out of moves, keep exploring
    while (not env.game_over()) and (not env.victory()) and (env.get_moves() < simLength):
        """
        INIT. DEFAULT POLICY: explore a random action from the list of available actions.
        Once an action is explored, remove from the available actions list
        """
        # Select a random action from this state
        actions = env.get_valid_actions()
        index = random.randint(0, len(actions))
        act = actions[index-1]

        # Take that action, updating env to a new state
        env.step(act)
        outcome = env.get_score()
        if(outcome > prevScore):
            return outcome

    #return the reward received by reaching terminal state
    return env.get_score()

"""
This function backpropogates the results of the Monte Carlo Simulation back up the tree

Keyword arguments:
node -- the child node we simulated from
delta -- the component of the reward vector associated with the current player at node v
"""
def backUp(node, delta):
    while node is not None:
        # Increment the number of times the node has 
        # been visited and the simulated value of the node
        node.visited += 1
        node.simValue += delta
        # Traverse up the tree
        node = node.getParent()



def selectAction(parent, exploration):
    max = -inf
    bestChild = None
    for child in parent.getChildren():
        # if there is a child we haven't explored, explore it
        if(child.visited != 0):
            childValue = (child.simValue/child.visited)
            # if there is a tie for best child, randomly keep one
            if(childValue == max and random.random() > .5):
                bestChild = child
                max = childValue
            #if it's value is greater than the best so far, it will be our best so far
            elif(childValue > max):
                bestChild = child
                max = childValue
    return bestChild





# A node in the MCTS
class Node:
    """
    This is a temporary class to be filled in later. It holds a state of the game.
    """

    def __init__(self, parent, prevAct):
        self.parent = parent
        self.prevAct = prevAct
        self.children = []
        self.simValue = 0
        self.visited = 0
        self.terminal = False

    def print(self, level):
        space = ""
        for i in range(level):
            space += ">"
        if(self.prevAct is None): 
            print("\t"+space+"<root>"+"\n")
        else:
            print("\t"+space+self.prevAct+"\n")

        for child in self.children:
            child.print(level+1)

    def addChild(self, child):
        self.children.append(child)

    def getParent(self):
        return self.parent

    def getPrevAction(self):
        return self.prevAct

    def getChildren(self):
        return self.children

    def isExpanded(self):
        if len(self.children) == 0 and not self.terminal:
            return False
        else:
            return True



############################### STUFF TO RUN DOWN SIMULATION BELOW

def reward(curr, terminal):
    """
    The reward method calculates the change in the score of the game from
    the current state to the end of the simulation.
    """
    return terminal.get_score()-curr

#######################

   


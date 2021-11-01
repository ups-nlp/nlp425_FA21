
from math import inf, sqrt, log2
import random




# Expand From the given node
def expandNode(parent):
    # Get ready to store the selected action
    action = ""

    # Find the best action
    for action in Node.actions: ## Node store its possible actions? or rather actions as method parameter
        action = "" ## Somehow select what action to do

    # Make a new node
    newNode = Node(parent, action, parent.actions) ## Node knows what action led to this state? as argument for constructor?

    ## Add newNode as child node of parent

    # Return the newly created node to-be-explored
    return newNode

def defaultPolicy(state):
    """
    The defaultPolicy represents a simulated exploration of the tree from
    the passed-in node to a terminal state. 
    """
    explored = [] #track all explored actions from the current state
    curr = state
    x = 0
    DEPTH = 10 #change this value depending on how far down the tree to search
    while x<= 10:
        """
        INIT. DEFAULT POLICY: explore a random action from the list of available actions.
        Once an action is explored, remove from the available actions list
        """
        index = random.randint(0, len(curr.actions))
        act = curr.actions[index]
        explored.append(act)
        curr.actions.remove(act)

        #create a new state resulting from taking action in current state
        """
        ERROR: to expand on any state, won't we need the possible actions for that state?
        And isn't this determined by the feedback from the game in response fo our action,
         which we won't have access to before actually taking that action.
         Currently, I am sending forward the list of remaining unexplored actions, but this is not valid
        """
        curr = Node(curr, act, curr.actions)
        x+=1
    #store the reward received by reaching terminal state
    rew = reward(curr)

    #reset simulation to current state (add back actions)
    while curr != state:
        temp = curr.parent
        curr = curr.remove()
        temp.actions.append(explored[len(explored)-1])
        explored.remove(explored[len(explored)-1])
        curr = temp
    #return the reward
    return rew

def reward(state):
    """
    The reward method will calculate the reward of any state given a reward function we will determine later.
    Currently, it just returns the amount of possible actions remaining.
    """
    return len(state.actions)

#######################

""" Travel down the tree to the ideal node to expand on

This function loops down the tree until it finds a 
node whose children have not been fully explored, or it 
explores the best child node of the current node. 

Keyword arguments:
root -- the root node of the tree
Return: the ideal node to expand on """
def treePolicy(root):
    node = root
    while node is not None:
        #if parent is not full expanded, expand it and return
        if(not node.isExpanded()):
            return expandNode(node)
        #Otherwise, look at the parent's best child
        else:
            # This constant balances tree exploration with exploitation of ideal nodes
            Exploration_Exploitation = 2
            # Select the best child of the current node to explore
            node = bestChild(node, Exploration_Exploitation)

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
    max = 0
    bestChild = None
    for child in parent.getChildren():
        # if there is a child we haven't explored, explore it
        if(child.visit == 0):
            return child
        # otherwise, check if this is the best child so far
        else:
            # Use the Upper Confidence Bounds for Trees to determine the value for the child
            childValue = (child.simValue/child.visit) + 2*exploration*sqrt((2*log2(parent.visited))/child.visited)
            #if it's calue is greater than the best so far, it will be our best so far
            if(childValue >= max):
                bestChild = child
                max = childValue
    return bestChild


# A node in the MCTS
class Node:
    """
    This is a temporary class to be filled in later. It holds a state of the game.
    """

    def __init__(self, parent, prevAct, actions):
        self.parent = parent
        self.prevAct = prevAct
        self.actions = actions
        self.children = []
        self.simValue = inf
        self.visited = 0

    def addChild(self, child):
        self.children.append(child)

    def getParent(self):
        return self.parent

    def getChildren(self):
        return self.children

    def isExpanded(self):
        for child in self.getChildren():
            # if there is a child we haven't explored, explore it
            if(child.visit == 0):
                return False
        else:
            return True




    """
    The remove method kills off a node and all of its children recursively.
    It removes itself from its parent's list of children, then sets itself equal to null.
    This function is used to kill nodes created in a simulation.
    """
    def remove(self):
        for chil in self.children:
            chil.remove()
        self.getParent().children.remove(self)
        return None


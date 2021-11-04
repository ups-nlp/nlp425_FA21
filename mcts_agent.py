
from math import inf, sqrt, log2, trunc
from jericho import FrotzEnv
import random



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

    #There is a case we need to figure out where there are no valid 
    # actions from the given state (If we have won/lost)
    
    # Create all possible child nodes
    for action in actions: 
        # Make a new node
        newNode = Node(parent, action, parent.depth+1) 
        parent.addChild(newNode)

    # update the env variable to the new node we are exploring
    env.step(newNode.getPrevAction())
    # Return a newly created node to-be-explored
    return newNode

""" 

Keyword arguments:
state -- the node being expanded
env -- FrotzEnv interface between the learning agent and the game (it is at the current state of the game)
Return: 
"""
def defaultPolicy(state, env, depthLimit):
    """
    The defaultPolicy represents a simulated exploration of the tree from
    the passed-in node to a terminal state. 
    """
    #if currently unexplored node, set score to 0
    if state.simValue == inf:
        state.simvalue = 0

    currScore = env.get_score()
    #track all explored actions from the current state
    explored = [] 
    curr = state
    x = 0
    DEPTH = 10 #change this value depending on how far down the tree to search
    while not (curr.isTerminal(env, curr, (state.depth+x),depthLimit)):
        """
        INIT. DEFAULT POLICY: explore a random action from the list of available actions.
        Once an action is explored, remove from the available actions list
        """
        index = random.randint(0, len(curr.actions))
        act = curr.actions[index]
        explored.append(act)
        curr.actions.remove(act)

        #create a new state resulting from taking action in current state
        curr = env.step(act)
        x+=1
    #store the reward received by reaching terminal state
    rew = reward(currScore,env)

    #reset simulation to current state (add back actions)
    while curr != state:
        temp = curr.parent
        curr = curr.remove()
        temp.actions.append(explored[len(explored)-1])
        explored.remove(explored[len(explored)-1])
        curr = temp
    #return the reward
    return rew


def defaultPolicy2(newNode, env, depthLimit):
    """
    The defaultPolicy represents a simulated exploration of the tree from
    the passed-in node to a terminal state. 

    Self-note: This method doesn't require the nodes to store their depth
    """
    #if currently unexplored node, set score to 0
    newNode.simvalue = 0

    prevScore = env.get_score()

    maxDepth = 10
    
    # While the game is not over and we have not run out of moves, keep exploring
    while (not env.game_over()) and (not env.victory()) and (env.get_moves() < maxDepth):
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
    
    # Potential error here when determining the score? All the nodes have negative scores

    #return the reward received by reaching terminal state
    outcome = env.get_score() - prevScore
    if outcome > 0:
        return outcome
    else:
        return 0

def reward(curr, terminal):
    """
    The reward method calculates the change in the score of the game from
    the current state to the end of the simulation.
    """
    return terminal.get_score()-curr

#######################

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
            node = bestChild(node, explore_exploit_const)
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
        # otherwise, check if this is the best child so far
        else:
            # Use the Upper Confidence Bounds for Trees to determine the value for the child
            childValue = (child.simValue/child.visited) + 2*exploration*sqrt((2*log2(parent.visited))/child.visited)
            #if it's calue is greater than the best so far, it will be our best so far
            if(childValue >= max):
                bestChild = child
                max = childValue
    return bestChild

def selectAction(parent, exploration):
    max = -inf
    bestChild = None
    for child in parent.getChildren():
        # if there is a child we haven't explored, explore it
        if(child.visited != 0):
            # Use the Upper Confidence Bounds for Trees to determine the value for the child
            childValue = (child.simValue/child.visited)
            #if it's calue is greater than the best so far, it will be our best so far
            if(childValue >= max):
                bestChild = child
                max = childValue
    return bestChild

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



# A node in the MCTS
class Node:
    """
    This is a temporary class to be filled in later. It holds a state of the game.
    """

    def __init__(self, parent, prevAct, depth):
        self.parent = parent
        self.prevAct = prevAct
        self.children = []
        self.simValue = 0
        self.visited = 0
        self.depth = depth

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
        if len(self.children) == 0:
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

############################### STUFF TO RUN DOWN SIMULATION BELOW

"""
Simulates the end state of the given path

This function creates a jericho environement and runs a game to a certain point getting the 'end state' for that line of actions

Keyword arguments:
path -- an array containing the actions used to get to this point
game_file -- the file of the game being played
stepLimit -- the max number of steps we want to simulate
Return: the endstate tuple generated after entering all the given actions (i think observation is the first item of the tuple)
"""
def simulatePath(path, game_file, stepLimit):
    # Create the environment
    env = FrotzEnv(game_file)

    # Make sure its reset just in case
    curState = env.reset()

    # Take steps except the last step
    for i in range(len(path)):
        # Check to see if we are in a terminal state
        if(isTerminal(curState, i, stepLimit)):
            # If we are at a terminal state return the last current state
            return curState
        else:
            # Take the action at this point in the path and store state tuple
            curState = env.step(path[i])

    # Return endstate of this path
    return curState

   
"""
Checks to see if this step of the simulation should be a terminal state

This function checks an environment and sees if it is at its step limit, a game over, or a successful end game terminal state

Keyword arguments:
env -- the current game environment
curState -- the current state of the game
curSteps -- the current number of steps taken
Return: true if the step is terminal, false if not
"""
def isTerminal(env, curState, curSteps, stepLimit):
    # Check if step limit reached
    if(curSteps > stepLimit):
        return True
    elif(curState[2]): # Check if the 'done' value of the game state is true
        return True
    else: # See if we are game over or not
        return env.game_over()
        

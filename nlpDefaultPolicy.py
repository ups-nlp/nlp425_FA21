import random

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

class Node:
    """
    This is a temporary class to be filled in later. It holds a state of the game.
    """

    def __init__(self, parent, prevAct, actions):
        self.parent = parent
        self.prevAct = prevAct
        self.actions = actions
        self.children = []

    def addChild(self, child):
        self.children.append(child)

    def getParent(self):
        return self.parent

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


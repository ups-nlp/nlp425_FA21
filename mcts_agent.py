
# Expand From the given node
def expandNode(parent):
    # Get ready to store the selected action
    action = ""

    # Find the best action
    for action in node.actions: ## Node store its possible actions? or rather actions as method parameter
        action = "" ## Somehow select what action to do

    # Make a new node
    newNode = node() ## Node knows what action led to this state? as argument for constructor?

    ## Add newNode as child node of parent

    # Return the newly created node to-be-explored
    return newNode


#######################


# A node in the MCTS
class node():
    print("PLACEHOLDER")
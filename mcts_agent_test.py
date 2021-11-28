from agent import mcts_agent
from jericho import FrotzEnv
from math import inf, sqrt, log2, floor,e
import random

class Test:
    # Create the environment
    env = FrotzEnv(game_file)
    actions = ["n","e","s","w"]
    root = mcts_agent.Node(None, None, actions)



    def test_node(self,root):
        assert self.root.get_parent() is None
        assert self.root.is_terminal()==0
        new_actions = []
        node1 = mcts_agent.Node(root, "n", new_actions)
        node2 = mcts_agent.Node(root, "e", new_actions)
        node3 = mcts_agent.Node(root, "s", new_actions)
        node4 = mcts_agent.Node(root, "w", new_actions)
        self.root.add_child(node1)
        self.root.add_child(node2)
        self.root.add_child(node3)
        self.root.add_child(node4)
        assert self.root.is_expanded()==1
        assert self.root.get_children() == [node1,node2,node3,node4]

    def test_expand_child(self, actions):
        assert mcts_agent.expand_node(self.root, self.env).get_prev_action() in actions # just checks to make sure that the action that was expanded existed in the list of possible actions

    def test_next_child(self, actions):
        new_actions = []

        general_score_min = 0.0 # a minimum score for the generic actions available 
        general_score_max = 5.0 # a maximum score for the generic actions available

        general_visited_min = 0 # a minimum score for the generic actions available 
        general_visited_max = 5 # a maximum score for the generic actions available

        random_act = floor(random(0, len(actions))) # a random action from the list
        high_val_score = 1.0 # the value for the random action chosen above ^^^^ that SHOULD be the one chosen
        high_val_visited = 1 # the times visited for the random action chosen above ^^^^ that SHOULD be the one chosen


        for i in range(0, len(actions)):
            new_node = mcts_agent.Node(self.root, actions[i], new_actions)
            
            if(i == random_act): # if this is the randomly selected 'good node', we give it the scores that should get it chosen
                new_node.sim_value = high_val_score
                new_node.visited = high_val_visited
                prospective_best = new_node # store the 'proposed best' node
            else: # if it is not the randomly selected node 'good node', give it random scores between the specified values
                new_node.sim_value = random(general_score_min, general_score_max)
                new_node.visited = floor(random(general_visited_min, general_visited_max))

            self.root.add_child(new_node) # add the node as a child of the root

        # Exploration exploitation constant
        exploration = 1.0

        # The reward policy used to calculate scores
        reward_policy = mcts_agent.Additive_Reward

        # check to see if the node returned is the node we think
        assert mcts_agent.best_child(self.root, exploration, self.env, reward_policy) == prospective_best

    def test_softmax():
        #test the return value for the softmac reward function
        assert floor(mcts_agent.Softmax_Reward.softmax_calc(0,10)) == 34843
        raise NotImplementedError
    
    def test_generalized_softmax():
        #test the return value for the generalized softmax reward function
        raise NotImplementedError

    def test_additive():
        #test the return value for the additive reward function
        raise NotImplementedError


        

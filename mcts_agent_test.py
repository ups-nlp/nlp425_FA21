from agent import mcts_agent
from jericho import FrotzEnv
from math import inf, sqrt, log2, floor,e
class Test:
    # Create the environment
    env = FrotzEnv(game_file)
    root = mcts_agent.Node(None, None, ["n","e","s","w"])

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


        

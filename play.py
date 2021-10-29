"""Instantiates an AI agent to play the specified game"""

import argparse

from jericho import FrotzEnv
from agent import Agent
from agent import RandomAgent
from agent import HumanAgent
from DEPagent import DEPagent


def play_game(agent: Agent, game_file: str, num_steps: int):
    """ The method that runs the game play. """

    # Create the environment
    env = FrotzEnv(game_file)

    # The history is a list of (observation, action) tuples
    history = []

    # Get the initial observation and info
    # info is a dictionary (i.e. hashmap) of {'moves':int, 'score':int}
    curr_obs, info = env.reset()
    done = False

    print('=========================================')
    print("Initial Observation\n" + curr_obs)

    while num_steps > 0 and not done:
        
        # For each step of game play, the agent determines the next action
        # based on env and the history of observations and actions
        # env is the environment from Frotz
        action_to_take = agent.take_action(env, history)
        
        # Take the action, which produces a new observation and updated info
        # done is true wh
        next_obs, _, done, info = env.step(action_to_take)

        history.append((curr_obs, action_to_take))

        curr_obs = next_obs

        print('\n\n=========================================')
        print('Taking action: ', action_to_take)
        print('Game State:', next_obs.strip())
        print('Total Score', info['score'], 'Moves', info['moves'])

        num_steps -= 1

    print('\n\n============= HISTORY OF ACTIONS TAKEN =============')
    for _, action in history:
        print(action)


def main(agent, game_file, num_moves):
    """ 
    The main method that instantiates an agent and calls play_game for the
    specified game.
    """

    
    # Right now all you can create is a RandomAgent or a HumanAgent. 
    # This will expand in the future.
    if agent == 'random':
        ai_agent = RandomAgent()
    elif agent == 'human':
        ai_agent = HumanAgent()
    elif agent == 'DEPagent':
        ai_agent = DEPagent()
    else:
        ai_agent = RandomAgent()

    play_game(ai_agent, game_file, num_moves)
    

if __name__ == "__main__":
    """
    Read in command line arguments and begin the game play.
    Use a parser for the command line arguments
    num_moves -- The number of moves the agent should make
    agent -- Right now this is just 'random' but will expand as we make 
    other agents
    game_file -- The full path to the game file
    """
    parser = argparse.ArgumentParser(
        description='Runs an AI agent on a specified game')

    # Get the argements that were read in from the command line
    parser.add_argument(
        'num_moves', type=int, help='Number of moves for the agent to make')
    parser.add_argument('agent', help='[random|human]')
    parser.add_argument('game_file', help='Full pathname for game')
    args = parser.parse_args()
    
    # Call the main code to set up the agent and start the game
    main(args.agent, args.game_file, args.num_moves)


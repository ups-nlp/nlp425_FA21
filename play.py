"""Instantiates an AI agent to play the specified game"""

import argparse

from jericho import FrotzEnv
from agent import Agent
from agent import RandomAgent
from agent import HumanAgent

import config

def play_game(agent: Agent, game_file: str, num_steps: int):
    """ The main method that instantiates an agent and plays the specified game"""

    # Create the environment
    env = FrotzEnv(game_file)

    # The history is a list of (observation, action) tuples
    history = []

    curr_obs, info = env.reset()
    done = False

    if config.VERBOSITY > 0:
        print('=========================================')
        print("Initial Observation\n" + curr_obs)

    while num_steps > 0 and not done:
        action_to_take = agent.take_action(env, history)

        # info is a dictionary (i.e. hashmap) of {'moves':int, 'score':int}
        next_obs, _, done, info = env.step(action_to_take)

        history.append((curr_obs, action_to_take))

        curr_obs = next_obs

        if config.VERBOSITY > 0:
            print('\n\n=========================================')
            print('Taking action: ', action_to_take)
            print('Game State:', next_obs.strip())
            print('Total Score', info['score'], 'Moves', info['moves'])

        num_steps -= 1

    if config.VERBOSITY > 1:
        print('\n\n============= HISTORY OF ACTIONS TAKEN =============')
        for _, action in history:
            print(action)

    return (info['score'],info['moves'])

if __name__ == "__main__":
    # Read in command line arguments and play the game with the specified parameters
    # Uses a parser for the command line arguments:
    # num_moves -- The number of moves the agent should make
    # agent -- Right now this is just 'random' but will expand as we make other agents
    # game_file -- The full path to the game file

    parser = argparse.ArgumentParser(
        description='Runs an AI agent on a specified game')

    parser.add_argument(
        'num_moves', type=int, help='Number of moves for the agent to make')
    parser.add_argument('agent', help='[random|human]')
    parser.add_argument('game_file', help='Full pathname for game')
    parser.add_argument('-v', '--verbosity', type=int, help='[0|1] verbosity level')
    args = parser.parse_args()

    # Right now all you can create is a RandomAgent. This will expand in the future
    if args.agent == 'random':
        ai_agent = RandomAgent()
    elif args.agent == 'human':
        ai_agent = HumanAgent()
    else:
        ai_agent = RandomAgent()

    # Set the verbosity level
    if args.verbosity == 0 or args.verbosity == 1:
        config.VERBOSITY = args.verbosity

    play_game(ai_agent, args.game_file, args.num_moves)

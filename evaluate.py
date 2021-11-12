""" Subroutines for evaluating agents """
import argparse

from jericho import FrotzEnv
from agent import Agent
from agent import RandomAgent
from agent import HumanAgent

from play import play_game

import config


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Evaluates an agent')

    parser.add_argument('num_trials', type=int, help='Number of times to run the agent on the specified game')
    parser.add_argument('num_moves', type=int, help='Number of moves for agent to take per trial')
    parser.add_argument('agent', help='[random|human]')
    parser.add_argument('game_file', help='Full pathname to the game file')
    parser.add_argument('-v', '--verbosity', type=int, help='[0|1] verbosity level')
    args = parser.parse_args()

    if args.agent == 'random':
        ai_agent = RandomAgent()
    elif args.agent == 'human':
        ai_agent = HumanAgent()
    else:
        ai_agent = RandomAgent()

    # Set the verbosity level
    if args.verbosity == 0 or args.verbosity == 1:
        config.verbosity = args.verbosity

    avg_score = 0
    for i in range(args.num_trials):            
        score, moves = play_game(ai_agent, args.game_file, args.num_moves)
        avg_score += score
        if config.verbosity > 0:
            print(f'Trial {i}:')
            print(f'final score={score}\ntotal moves={moves}\n')        
        
    print()
    print(f'Average score: {avg_score/args.num_trials}')


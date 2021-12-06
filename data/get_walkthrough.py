#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 15:02:22 2021

@author: prowe

A test player who gets us manually to a certain part of the game
in order to test things
"""

# Built-in modules
import csv

# Installed third-party modules
from jericho import FrotzEnv

# Our modules
import config

def set_up_game(game_file: str):
    # Create the environment
    env = FrotzEnv(game_file)

    # The history is a list of (observation, action) tuples
    history = []

    # Get the initial observation and info
    # info is a dictionary (i.e. hashmap) of {'moves':int, 'score':int}
    curr_obs, info = env.reset()
    
    return curr_obs, history, env


    
def game_step(action_to_take, curr_obs, history, env):
    history.append((curr_obs, action_to_take))
    next_obs, _, done, info = env.step(action_to_take)
    curr_obs = next_obs

    if config.VERBOSITY > 0:
        print('\n\n=========================================')
        print('Taking action: ', action_to_take)
        print('Game State:', next_obs.strip())
        print('Total Score', info['score'], 'Moves', info['moves'])
    
    return curr_obs, history, env


def print_status(env, history):
    print_history(history)
    print_inventory(env)
    
def print_inventory(env):
    print('\n\n============= INVENTORY =============')
    for item in env.get_inventory():
        print(item.name)
        
def print_history(history):
    print('\n\n============= HISTORY  =============')
    for obs, action in history:
        print(obs, "   ---> Action:", action)

def print_actions(history):
    print('\n\n============= HISTORY OF ACTIONS TAKEN =============')
    for _, action in history:
        print(action)



    
def get_walkthrough(game_file, walkthrough_source):
    """
    Play the game for a walkthrough and get the accompanying
    observations
    """
    
    
    # Set up and play the game
    curr_obs, history, env = set_up_game(game_file)
    
    # Get the actions - this could be, e.g. from a file. Here it is from
    # the built-in walkthrough
    if walkthrough_source == 'builtin':
        actions = env.get_walkthrough()
    else:
        raise ValueError('No other walkthroughs available yet')

    walkthrough = []
    # Take the actions, one by one
    while len(actions) > 0:
        prev_score = env.get_score()
        action_to_take = actions.pop(0)
        curr_obs, history, env = game_step(action_to_take, curr_obs, 
                                           history, env)
        points = env.get_score() - prev_score
        walkthrough.append((history[-1][0], history[-1][1], points))

    if config.VERBOSITY > 0:
        print_actions(history)
        print_history(history)
        print_inventory(env)
    
    print('\nConfirm I won:', env.victory())
    
    return walkthrough




if __name__ == "__main__":
    """ 
    Create a walkthrough as a csv file of observation, action
    for a given file with a list of actions, or the builtin walkthrough
    
    """
    # # # # # # # # # # # #   INPUTS    # # # # # # # # # # # # # # # #
    # The location of your Jericho game suite. This is unique to your computer,
    # so you should modify it
    jericho_dir = '/Users/prowe/software/jericho-master/z-machine-games-master/' 
                  
    # The name of the game you want to play
    game_file = jericho_dir + 'jericho-game-suite/zork1.z5'
        
    # The source of the actions
    walkthrough_source = 'builtin'

    # The file to save to
    outfile = 'data/frotz_builtin_walkthrough.csv'
    # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    

    # Get a list of tuples of observation, action    
    walkthrough = get_walkthrough(game_file, walkthrough_source)
    
    # Write the file
    with open(outfile,'w') as out:
        csv_out = csv.writer(out)
        csv_out.writerow(['Observation', 'Action', 'Points'])
        for row in walkthrough:
            csv_out.writerow(row)



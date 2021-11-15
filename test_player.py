#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 15:02:22 2021

@author: prowe

A test player who gets us manually to a certain part of the game
in order to test things
"""


from jericho import FrotzEnv
from agent import Agent
from dep_agent import DEPagent


def set_up_game(agent: Agent, game_file: str):
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



""" Set up for the test """
# The location of your Jericho game suite. This is unique to your computer,
# so you should modify it
jericho_dir = '/Users/prowe/software/jericho-master/z-machine-games-master/' \
              + 'jericho-game-suite/'
              
# The name of the game you want to play
game = 'zork1.z5'
game_file = jericho_dir + game

# The type of agent: random, human, DEP, etc
ai_agent = DEPagent()

# The number of moves
num_moves = 100

# Set up the game
curr_obs, history, env = set_up_game(ai_agent, game_file)


# Test of hoarder - keep taking stuff until we fill up the inventory
pre_actions = ['open mailbox', 'take leaflet', 'north','east','open window',
               'go in', 'take all', 'west', 'take all','east',
               'turn on lantern', 'up', 'take all', 'down', 'west',
               'move rug', 'open trap door','down','north']
#               'kill troll with sword', 'take all','drop leaflet',
#               'drop brown sack', 'take axe']

# When we have too much stuff (7-8 items), we get the message:
# "Your load is too heavy"
# But it depends on the weight of the item to be picked up!


# Take the prescribed actions
while len(pre_actions) > 0:
    action_to_take = pre_actions.pop(0)
    curr_obs, history, env = game_step(action_to_take, curr_obs, history, env)
 

# Let the agent take over
while num_moves > 0:
    num_moves-=1
    action_to_take = ai_agent.take_action(env, history)
    curr_obs, history, env = game_step(action_to_take, curr_obs, history, env)
    # print_status(env, history)

action_to_take = ai_agent.take_action(env, history)

print_actions(history)
print_history(history)
print_inventory(env)


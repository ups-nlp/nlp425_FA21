#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 19:53:30 2021

@author: prowe

This is a driver for calling the main code in play.py to execute the
text-based game AI. Use this if you don't want to, or can't call play.py
from the command line via, e.g.

python play.py 10 random /Users/prowe/software/jericho-master/z-machine-games-master/jericho-game-suite/zork1.z5

"""

from play import play_game
from DEPagent import DEPagent
from agent import RandomAgent
from agent import HumanAgent

# The location of your Jericho game suite. This is unique to your computer,
# so you should modify it
jericho_dir = '/Users/prowe/software/jericho-master/z-machine-games-master/' \
              + 'jericho-game-suite/'
              
# The name of the game you want to play
game = 'zork1.z5'
game_file = jericho_dir + game

# The type of agent: random, human, DEP, etc
agent = "DEPagent"

# The number of moves
num_moves = 10



# Instantiate an agent 
if agent == 'random':
    ai_agent = RandomAgent()
elif agent == 'human':
    ai_agent = HumanAgent()
elif agent == 'DEPagent':
    ai_agent = DEPagent()
else:
    ai_agent = RandomAgent()


# call play_game for the specified game.
play_game(ai_agent, game_file, num_moves)


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 19:53:30 2021

@author: prowe

This is a driver for calling the main code in play.py to execute the
text-based game AI. Use this if you don't want to, or can't call play.py
from the command line
"""

import play

# The location of your Jericho game suite. This is unique to your computer,
# so you should modify it
jericho_dir = '/Users/prowe/software/jericho-master/z-machine-games-master/' \
              + 'jericho-game-suite/'
              
# The name of the game you want to play
game = 'zork1.z5'

# The type of agent: random, human, DEP, etc
agent = "DEP"

# The number of moves
num_moves = 10

play.main(agent, jericho_dir + game, num_moves);

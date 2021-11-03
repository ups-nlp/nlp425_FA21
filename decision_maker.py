#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 14:34:43 2021

@author: 
"""

import random

def decision_maker(valid_actions:list, history:list) -> int:
    """ 
    Decide which choice to take randomly for now
    this needs some intelligence
    
    @param valid_actions
    @param history
    
    Creates an embedding of all the words in the previous observation,
    runs that through a neural network that ranks how much we should use 
    each of the modules. Then returns an int that represents the module
    with the highest value that has valid actions
    
    @return chosen_module: an integer between 0 and 3, representing the
                           action module we are going to use
    """
    chosen_module = random.randint(0, 3)
    return chosen_module

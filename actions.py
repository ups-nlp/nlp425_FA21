@author brayancodes
# Imports from my jupyter notebook.
from collections import Counter
import math
import random
import numpy as np
import os
import re
import statistics
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import spacy
from spacy import displacy

# https://pypi.org/project/benepar/
import benepar
benepar.download('benepar_en3')

nlp = spacy.load('en_core_web_lg')

if spacy.__version__.startswith('2'):
    nlp.add_pipe(benepar.BeneparComponent("benepar_en3"))
else:
    nlp.add_pipe("benepar", config={"model": "benepar_en3"})

# Test Input from Zork 1
input_1 = "You are facing the south side of a white house. There is no door here, and all the windows are boarded."
input_2 = "You are behind the white house. A path leads into the forest to the east. In one corner of the house there is a small window which is slightly ajar."
input_3 = "You are in the kitchen of the white house. A table seems to have been used recently for the preparation of food. A passage leads to the west and a dark staircase can be seen leading upward. A dark chimney leads down and to the east is a small window which is open."
input_4 = "You are in the living room. There is a doorway to the east, a wooden door with strange gothic lettering to the west, which appears to be nailed shut, a trophy case, and a large oriental rug in the center of the room."
input_5 = "With a great effort, the rug is moved to one side of the room, revealing the dusty cover of a closed trap door."

# Prompt for Input
user_input = input("Enter 1 - 5 to choose a different sentences to parse from Zork 1: ")
if user_input == 1:
    desired_input = input_1
elif user_input == 2:
    desired_input = input_2
elif user_input == 3:
    desired_input = input_3
elif user_input == 4:
    desired_input = input_4
else:
    desired_input = input_5
print()
print("You have chosen this sentence:")
print(desired_input)
print()

# Parse the desired input &kioi print
doc = nlp(desired_input)
sentences = (doc.sents)

for sentence in sentences:
    print()
    print(sentence._.parse_string)
    print()

# Parse the actual nouns from the Spacy output



# The Standard Function for the Final Project
get.actions():
print("Hello")
#@author brayancodes, real-froggy-chair
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
# Commenting this out as it's already cached on our system
#benepar.download('benepar_en3')

nlp = spacy.load('en_core_web_lg')

if spacy.__version__.startswith('2'):
    nlp.add_pipe(benepar.BeneparComponent("benepar_en3"))
else:
    nlp.add_pipe("benepar", config={"model": "benepar_en3"})

def get_directions(input_string):
	#Extremely rudimentary dictionary containing direction info 
	directionDict = {
		"north" : {'north', 'front', 'ahead'},
		"south" : {'south', 'behind', 'back'},
		"east"  : {'east', 'right'},
		"west"  : {'west', 'left'}
	}

	working_strings = input_string.split('\n')

	for s in working_strings:
		doc = nlp(s) #run information from game through the nlp pipeline
		text = [token.text for token in doc]
		pos  = [token.pos_ for token in doc]
		sent = list(doc.sents)[0]

		print("POS: ", pos)
		print("TEXT: ", text)

		# Using input of enviornment, determine directional words
		parsed = sent._.parse_string

		# Plan for later:
		# Read parse tree to understand relationship between direction and object
		# E.x. differentiate "West of house" means going east goes to house
		# v.s. "to the north" means going north follows that path 

if __name__ == "__main__" :
	#Things to be run from commandline 
	env = FrotzEnv(game_file)
	info = env.reset()[0]

	#cleans out the copyright info for pipeline
	info = info.split('\n', maxsplit = 4)[-1].strip()

	get_directions(info)
    get_nouns(info)

# Test Input from Zork 1
input_1 = "You are facing the south side of a white house. There is no door here, and all the windows are boarded."
input_2 = "You are behind the white house. A path leads into the forest to the east. In one corner of the house there is a small window which is slightly ajar."
input_3 = "You are in the kitchen of the white house. A table seems to have been used recently for the preparation of food. A passage leads to the west and a dark staircase can be seen leading upward. A dark chimney leads down and to the east is a small window which is open."
input_4 = "You are in the living room. There is a doorway to the east, a wooden door with strange gothic lettering to the west, which appears to be nailed shut, a trophy case, and a large oriental rug in the center of the room."
input_5 = "With a great effort, the rug is moved to one side of the room, revealing the dusty cover of a closed trap door."

# ===== get_nouns ======
def get_nouns(input):

    # Giving spacey the sentence
    doc = nlp(input)
    sentences = (doc.sents)

    text = [token.text for token in doc] # Words
    pos = [token.pos_ for token in doc] # Pos tags

    words_to_get = [] # This is storing the indexes for the words which are nouns.
    counter = 0
    for word in pos:# This is finding the indexes for the nouns.
        if word == 'NOUN':
            words_to_get.append(counter)
        counter += 1

    words_we_wanted = [] # This is storing the actual nouns
    for index in words_to_get: # Pulling the noun from its list using the found index.
        words_we_wanted.append(text[index])

    print()
    print(words_we_wanted)
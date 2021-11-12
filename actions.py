#@author brayancodes, real-froggy-chair
# Imports from my jupyter notebook.

from collections import Counter
import math
import random
import numpy as np
import os
import re
import statistics
import spacy
from jericho import FrotzEnv

nlp = spacy.load('en_core_web_lg')

def get_directions(input_string):
	#Extremely rudimentary dictionary containing direction info 
	directionDict = {
		"north" : {'north', 'front', 'ahead'},
		"south" : {'south', 'behind', 'back'},
		"east"  : {'east', 'right'},
		"west"  : {'west', 'left'}
	} 

	output = []
	doc = nlp(input_string.lower()) #run information from game through the nlp pipeline
	sentences = doc.sents

	for s in sentences:
		for token in s:
			if token.text in directionDict:

				#SPRINT 1
				direction = token.text
				output.append(direction)

	print(output)
	return(output)

			# SPRINT 2  
			# This builds a subtree around a directional word, allowing us to analyze that tree alone
			#subtree = token.subtree

			#if 'of' in subtree:
			#	match direction:
			#		case 'east':
			#			output.append('west')
			#		case 'west':
			#			output.append('east')
			#		case 'north':
			#			output.append('south')
			#		case 'south':
			#			output.append('north')
			#else
			#	output.append(direction)

			#print([t.text for t in subtree])
			#print(output)
		#elif token.pos_ is ("ADV" or "ADP"):
		#	subtree = token.subtree
		#	print([t.text for t in subtree])
		#	print([t.pos_ for t in subtree])

	# There should be a way to use the subtree we have to parse for modifier
	# E.g. "west of a white house", the "of" tells us the house is the subject and west is adverb
	# Using this we can decide that we need to go east to reach the house 
	# Can't just decide to go west because it says west 
		

	# Plan for later:
	# Read parse tree to understand relationship between direction and object
	# E.x. differentiate "West of house" means going east goes to house
	# v.s. "to the north" means going north follows that path 

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

	text = [token.text for token in doc]
	pos = [token.pos_ for token in doc]

	nouns = [] # This is storing the indexes for the words which are nouns.
	index = 0
	for word in pos:# This is finding the indexes for the nouns.
		if word == 'NOUN':
			nouns.append(text[index])
		index += 1
	return nouns

# The method that will be called when a list of valid actions is needed.
# @ game_observation - The current game observation.
# @ history - A list of all the game observations.
def get_valid_actions(game_observation, history):

	# Add get_directions once its been cleaned up & add  to the return
	list = get_nouns(game_observation)
	return list

# ====== Main Method =======
if __name__ == "__main__" :
	#Things to be run from commandline

	env = FrotzEnv('zork1.z5')
	info = env.reset()[0]

	#cleans out the copyright info for pipeline
	info = info.split('\n', maxsplit = 4)[-1].strip()

	get_directions(input_4)
	##get_nouns(info)
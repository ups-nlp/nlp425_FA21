# @author brayancodes, real-froggy-chair, Julien B.
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

def is_in_dict(token, dict):
	for key in dict.keys():
		for value in dict[key]:
			if token == value:
				return key
	return False

# This method creates a list of directions.
# @ input_string - The game observation.
# @ return - Returns a set of.
def get_directions(input_string):
	#Extremely rudimentary dictionary containing direction info 
	directionDict = {
		"north" : {'north', 'front', 'ahead'},
		"south" : {'south', 'behind', 'back'},
		"east"  : {'east', 'right'},
		"west"  : {'west', 'left'},
		"up"	: {'upward', 'upwards', 'upstairs', 'up'},
		"down"  : {'below', 'beneath', 'down'}
	} 

	output = []
	doc = nlp(input_string.lower()) #run information from game through the nlp pipeline
	sentences = (doc.sents)

	for s in sentences:
		s = s.as_doc() # this processes the sentence as a doc, so we can iterate through tokens

		for token in s:
			print(token.text, token.dep_, token.head.text, token.head.pos_,
            [child for child in token.children])

		for t in s: # for every token in the sentence
			dict_check = is_in_dict(t.text, directionDict)

			if dict_check is not False:
				direction = dict_check

				# SPRINT 2  
				# This builds a subtree around a directional word, allowing us to analyze that tree alone
				subtree = t.subtree
				p_st = [t.text for t in subtree] # iterable text version of subtree
				print(p_st)

				if 'of' in p_st:
					print("== of found")
					print("opposite of " + direction)
					if direction == 'east':
						output.append('west')
					elif direction == 'west':
						output.append('east')
					elif direction == 'north':
						output.append('south')
					elif direction == 'south':
						output.append('north')
				else:
					print("== no of")
					print(direction)
					output.append(direction)

				break # okay now move onto the next token  

			# Token is not in the dictionary, but we suspect it's directional
			# This has a lot of false positives. Need to refine it more 
			# Not entirely sure this is within scope of direction finder -- is "here" a direction?
			#elif t.pos_ is ("ADV" or "ADP") and not t.text in output: 
			#	print("== not in dict")
			#	print(t)
			#	print(t.pos_)
			#	print(t.lemma_)
			#	print([t.text for t in t.subtree])
			#	print("ROOT IS: " + [chunk.root for chunk in t.subtree.as_doc().nouns_chunks])
			#	output.append(t.text)						
	print(set(output))
	return set(output)

# This method creates a list of nouns based on the observation using spacy POS tagging.
# @ input - The game observation.
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

# This method will create actions phrases given a list of nouns and a list of verbs
# Return action_phrases - The list of valid actions.
def create_action_phrases(list_of_verbs, list_of_nouns, list_of_directions):
	action_phrases = []

	# Putting them together	
	for verb in list_of_verbs:
		for noun in list_of_nouns:
			phrase =  verb + " " +  noun
			action_phrases.append(phrase)

	# Include a standalone list of verbs
	for verb in list_of_verbs:
		action_phrases.append(verb)

	# Integrating Ben's directions here
	for direction in list_of_directions:
		action_phrases.append(direction)

	# Create a method that will implicity create an inventory for the player
	# This will require
	# 	To add to inventory: Check the last action taken and parse the first word
	#		to see if its take, if it is. Then add the second word to the noun list.
	#	To remove an item: Check if the last action taken's first word is drop. If
	#		if is. Then remove the second word  from the inventory noun list.

	return action_phrases

# The method that will be called when a list of valid actions is needed.
# This is being run from play.py, for now, so that we can get current observations.
# @ game_observation - The current game observation.
# @ history - A list of all the game observations.
def get_valid_actions(observation, gamefile):
	env = FrotzEnv(gamefile)

	# Making the action phrases
	action_phrases = create_action_phrases(get_verbs(env), get_nouns(observation), get_directions(observation))

	return action_phrases

# This method creates a list of verbs that will be used in to create action phrases.
# @ environment - The environment
def get_verbs(environment):
	walkthrough  = environment.get_walkthrough()
	verblist = []
	for x in walkthrough:
		newadd = x.split(' ')
		if newadd[0] not in ['N', 'Ne', 'Nw', 'S', 'Se', 'Sw', 'E', 'W', 'U', 'D'] and newadd[0].lower() not in verblist:
			verblist.append(newadd[0].lower())

	return verblist

# ====== Main Method =======
# NOT IN USE RIGHT NOW
if __name__ == "__main__" :
	
	#NOT IN USE RIGHT NOW

	env = FrotzEnv('zork1.z5')
	info = env.reset()[0]

	#cleans out the copyright info for pipeline
	info = info.split('\n', maxsplit = 4)[-1].strip()

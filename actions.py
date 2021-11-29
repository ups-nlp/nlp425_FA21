"""Replaces the get_valid_actions() call in the Frotz enviornment.

Subdivided into get_directions(), get_nouns(), and get_verbs(), which do what they say they do.
We then pair up all these into a giant potential actions list, that is processed through a probability
layer to determine if a given combo is possible (e.g. 'open forest' is not likely valid and is eliminated).

Authors: real-froggy-chair, brayancodes, Julien B.
Class: CS425 Fall 2021
Last working Compile Date: 11/28/2021"""

# TODO: eliminate unnecesary imports 
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
	"""Creates a set of predicted directionality for a given string input.

	Takes a given input string from the enviornment and runs it through Spacy
	to tokenize it. Tokenized string is then compared against a directionality dictionary
	to determine base direction (e.g. 'upwards' -> 'up') and outputs direciton as a set."""

	# Reading in directionality information from file to create hashmap
	directionDict = dict()

	with open('directions.txt', 'r') as f:
		lines = f.readlines()
		for line in lines:
			line = line.strip().split(' ')
			directionDict[line[0]] = line[1]
		f.close()
	#print(directionDict)

	output = []
	doc = nlp(input_string.lower()) # run information from game through the nlp pipeline
	sentences = (doc.sents)

	for s in sentences:
		s = s.as_doc() # this processes the sentence as a doc, so we can iterate through tokens

		#BUGTEST: Prints out current token information
		"""for token in s:
			print(token.text, token.dep_, token.head.text, token.head.pos_,
           [child for child in token.children])"""

		for t in s: 
			if t.text in directionDict:
				direction = directionDict[t.text]

				# SPRINT 2  
				# This builds a subtree around a directional word, allowing us to analyze that tree alone
				subtree = t.subtree
				p_st = [t.text for t in subtree] # iterable text version of subtree
				#print(p_st)

				if 'of' in p_st:
					#print("== of found")
					#print("opposite of " + direction)
					# This version of python does not have switches so using this
					if direction == 'east':
						output.append('west')
					elif direction == 'west':
						output.append('east')
					elif direction == 'north':
						output.append('south')
					elif direction == 'south':
						output.append('north')
				else:
					#print("== no of")
					#print(direction)
					output.append(direction)

				break # okay now move onto the next token  

			# Token is not in the dictionary, but we suspect it's directional
			# This has a lot of false positives. Need to refine it more 
			# Not entirely sure this is within scope of direction finder -- is "here" a direction?
			"""elif t.pos_ is ("ADV" or "ADP") and not t.text in output: 
				print("== not in dict")
				print(t)
				print(t.pos_)
				print(t.lemma_)
				print([t.text for t in t.subtree])
				print("ROOT IS: " + [chunk.root for chunk in t.subtree.as_doc().nouns_chunks])
				output.append(t.text)"""						
	#print(set(output))
	return set(output)

def get_nouns(input):
	'''This method creates a list of nouns based on the observation using spacey POS tagging.

	Keyword arguments:
	input -- The game observation

	Return:
	nouns -- The list of generated nouns from the game observation
	'''

	doc = nlp(input)
	sentences = (doc.sents)

	text = [token.text for token in doc]
	pos = [token.pos_ for token in doc]

	nouns = [] # This is storing the indexes for the words which are nouns.
	index = 0
	for word in pos:# This is finding the indexes for the nouns.
		if word == 'NOUN':
			# Remove cardinal directions from the list of nouns
			if text[index] != 'west' and text[index] != 'south' and text[index] != 'east' and text[index] != 'north':
				nouns.append(text[index])
		index += 1
	return nouns

def create_action_phrases(list_of_verbs, list_of_nouns, list_of_directions, inventory):
	'''This method will create the various types of action phrases.

	Keyword arguments:
	list_of_verbs -- A list of verbs to use for action phrases
	list_of_nouns -- A list of nouns to use for action phrases
	list_of_directions -- A list of directions to use for action phrases
	inventory -- A list of the player's inventory

	Return:
	action_phrases -- A list of valid action phrases
	'''
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

	# Use inventory items on nouns
	for item in inventory:
		for noun in list_of_nouns:
			phrase = "use " + item + " on " + noun
			action_phrases.append(phrase)

	return action_phrases

def get_valid_actions(observation, game_environment):
	'''This method will call create action phrases; called when a list of action phrases is needed.

	Keyword arguments:
	observation -- The current game observation
	game_environment -- The game environment being used in play.py

	Return:
	action_phrases -- A list of all the valid action phrases
	'''
	# Get the inventory
	inventory = game_environment.get_inventory() # This is a jericho object

	# Parse the inventory objects for the inventory items
	inventory_items = parse_inventory(inventory)

	# Making the action phrases
	action_phrases = create_action_phrases(get_verbs(game_environment), get_nouns(observation), get_directions(observation), inventory_items)

	return action_phrases

def parse_inventory(list_of_inventory_objs):
	'''This method turns inventory objects into a list of inventory item strings.

	Keyword arguments:
	list_of_inventory_objs -- A list of jericho inventory objects

	Return:
	list_of_items -- A list of strings representing the items in the players inventory
	'''

	list_of_items = []

	for item in list_of_inventory_objs:
		string = str(item) # obj -> string
		# Operations to parse the, now a string, inventory object.
		list_of_tokens = string.split("Parent")
		inventory_item = list_of_tokens[0].split(" ", 1)
		list_of_items.append(inventory_item[1][:-1])
	
	return list_of_items


def get_verbs(environment):
	'''This method will create a list of verbs based on the environment walkthrough.

	Keyword arguments:
	environment -- The game environment

	Return:
	verblist -- A list of valid verbs
	'''
	walkthrough  = environment.get_walkthrough()
	verblist = []
	for x in walkthrough:
		newadd = x.split(' ')
		if newadd[0] not in ['N', 'Ne', 'Nw', 'S', 'Se', 'Sw', 'E', 'W', 'U', 'D'] and newadd[0].lower() not in verblist:
			verblist.append(newadd[0].lower())

	return verblist

if __name__ == "__main__" :
	'''The main method. Currently not in use.
	'''
	#env = FrotzEnv('zork1.z5')
	#info = env.reset()[0]

	#cleans out the copyright info for pipeline
	#info = info.split('\n', maxsplit = 4)[-1].strip()

	#get_directions(info)

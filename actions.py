from jericho import FrotzEnv
import numpy as np
import spacy 
game_file = "/mnt/c/users/meani/Documents/cs425/GAME/z-machine-games-master/jericho-game-suite/zork1.z5"

nlp = spacy.load('en_core_web_sm')

#def get_valid_actions():
	# This must output a list
	# Brayan is working on this

def get_directions(input_string):
	doc = nlp(input_string) #run information from game through the nlp pipeline


	


	# Using input of enviornment, determine directional words


if __name__ == "__main__" :
	#Things to be run from commandline 
	env = FrotzEnv(game_file)
	info = env.reset()
	text = info[0]

	get_directions(text)
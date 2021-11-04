from jericho import FrotzEnv
import numpy as np
import spacy 
import benepar
game_file = "/mnt/c/users/meani/Documents/cs425/GAME/z-machine-games-master/jericho-game-suite/zork1.z5"


# Some init stuff for NLP and Benepar
nlp = spacy.load('en_core_web_sm')
benepar.download('benepar_en3')

if spacy.__version__.startswith('2'):
	nlp.add_pipe(benepar.BeneparComponent("benepar_en3"))
else:
	nlp.add_pipe("benepar", config={'model': 'benepar_en3'})

#def get_valid_actions():
	# This must output a list
	# Brayan is working on this

def get_directions(input_string):
	doc = nlp(input_string) #run information from game through the nlp pipeline
	text = [token.text for token in doc]
	pos  = [token.pos_ for token in doc]

	# Using input of enviornment, determine directional words
	print(doc._.parse_string)


if __name__ == "__main__" :
	#Things to be run from commandline 
	env = FrotzEnv(game_file)
	info = env.reset()
	text = info[0]

	get_directions(text)
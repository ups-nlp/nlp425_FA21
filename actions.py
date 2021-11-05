from jericho import FrotzEnv
import numpy as np
import spacy 
import benepar
game_file = "/mnt/c/users/meani/Documents/cs425/GAME/z-machine-games-master/jericho-game-suite/zork1.z5"


# Some init stuff for NLP and Benepar
nlp = spacy.load('en_core_web_sm')
# Uncomment this if you need to download, but it lags my interpreter 
# benepar.download('benepar_en3')

if spacy.__version__.startswith('2'):
	nlp.add_pipe(benepar.BeneparComponent("benepar_en3"))
else:
	nlp.add_pipe("benepar", config={'model': 'benepar_en3'})

#def get_valid_actions():
	# This must output a list
	# Brayan is working on this

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
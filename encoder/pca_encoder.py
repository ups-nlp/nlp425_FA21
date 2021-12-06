

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
import spacy
from sklearn.decomposition import PCA
import string


nlp = spacy.load('en_core_web_lg')



# The input file
INPUT_FILE = 'data/frotz_builtin_walkthrough.csv'

# Read in the data
df = pd.read_csv(INPUT_FILE)
observations = (df['Observation']).values    # The inputs
actions = (df['Action']).values              # The labels, or correct answers
points = (df['Points']).values               # The points earned by the action

# # # # # # # # # #      Clean and set up the data      # # # # # # # # # #
# Encode the observations
# The inputs are size: inputs x features: (396 x 768)
# The inputs vary from -1.2 to 1.4, so they are already pretty well
# centered and scaled, and they are already flat
# This encoder produces vectors of length 768: multi-qa-mpnet-base-dot-v1
# This one produces vectors of length 3xx: all-MiniLM-L6-v2
sentenceTransformer = SentenceTransformer('multi-qa-mpnet-base-dot-v1')
inputs = np.array([sentenceTransformer.encode([x])[0] for x in observations])


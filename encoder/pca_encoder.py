

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
#import spacy
from sklearn.decomposition import PCA
import string
from matplotlib import pyplot as plt


#nlp = spacy.load('en_core_web_lg')



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
pca = PCA(n_components=50)
proj_down = pca.fit_transform(inputs)


# Plot the inputs to see what they look like

ifig = 4
plt.figure(ifig); plt.clf()
plt.plot(np.arange(np.shape(inputs)[0]), inputs)
plt.xlabel('observation')
plt.ylabel('value')
plt.title('Each curve represents the values for a particular feature')
plt.show()
    
ifig += 1
plt.figure(ifig); plt.clf()
plt.plot(np.arange(np.shape(inputs)[1]), inputs.T)
plt.xlabel('feature')
plt.ylabel('value')
plt.title('Each curve represents the values for a particular observation')
    
    # Plot the encoded data, just to see what it looks like
ifig += 1
plt.figure(ifig); plt.clf()
plt.plot(np.arange(np.shape(proj_down)[0]), proj_down)
plt.xlabel('observation')
plt.ylabel('value')
plt.title('Each curve represents the values for a particular feature')
    
ifig += 1
plt.figure(ifig); plt.clf()
plt.plot(np.arange(np.shape(proj_down)[1]), proj_down.T, '.-')
plt.xlabel('feature')
plt.ylabel('value')
plt.title('Each curve represents the values for a particular observation')
"""
    # Plot a particular test case at index i
i = 0
ifig += 1
plt.figure(ifig); plt.clf()
plt.subplot(211)
plt.plot(inputs[i],label = 'Original')
plt.plot(proj_down[i], label = 'Decoded')
plt.legend()
plt.subplot(212)
plt.plot(inputs[i] - proj_down[i], label = 'original - decoded')
plt.legend()
    
ifig += 1
plt.figure(ifig)
plt.title('Original - Decoded')
for i in range(np.shape(proj_down)[0]):
    plt.plot(proj_down[i] - proj_down[i])
""" 
    
    


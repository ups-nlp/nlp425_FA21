

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt


def make_plots(df, st_encoded, prof_down):
    # Plot the inputs to see what they look like
    actions = (df['Action']).values              # The labels, or correct answers
    points = (df['Points']).values               # The points earned by the action

    ifig = 4
    plt.figure(ifig); plt.clf()
    plt.plot(np.arange(np.shape(st_encoded)[0]), st_encoded)
    plt.xlabel('observation')
    plt.ylabel('value')
    plt.title('Original: each curve represents the values for a particular feature')
    plt.show()
        
    ifig += 1
    plt.figure(ifig); plt.clf()
    plt.plot(np.arange(np.shape(st_encoded)[1]), st_encoded.T)
    plt.xlabel('feature')
    plt.ylabel('value')
    plt.title('Original: each curve represents the values for a particular observation')
        
        # Plot the encoded data, just to see what it looks like
    ifig += 1
    plt.figure(ifig); plt.clf()
    plt.plot(np.arange(np.shape(proj_down)[0]), proj_down)
    plt.xlabel('observation')
    plt.ylabel('value')
    plt.title('Encoded: each curve represents the values for a particular feature')
        
    ifig += 1
    plt.figure(ifig); plt.clf()
    plt.plot(np.arange(np.shape(proj_down)[1]), proj_down.T, '.-')
    plt.xlabel('feature')
    plt.ylabel('value')
    plt.title('Encoded: each curve represents the values for a particular observation')
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
        
    
def pca_encoder(INPUT_FILE):
    # The input file
    
    # Read in the data
    df = pd.read_csv(INPUT_FILE)
    observations = (df['Observation']).values    # The inputs
    
    # # # # # # # # 
    # #      Clean and set up the data      # # # # # # # # # #
    # Encode the observations
    # The inputs are size: inputs x features: (396 x 768)
    # The inputs vary from -1.2 to 1.4, so they are already pretty well
    # centered and scaled, and they are already flat
    # This encoder produces vectors of length 768: multi-qa-mpnet-base-dot-v1
    # This one produces vectors of length 3xx: all-MiniLM-L6-v2
    sentenceTransformer = SentenceTransformer('multi-qa-mpnet-base-dot-v1')
    st_encoded = np.array([sentenceTransformer.encode([x])[0] for x in observations])
    pca = PCA(n_components=50)
    proj_down = pca.fit_transform(st_encoded)
    
    return proj_down, st_encoded, df


if __name__ == "__main__":
    INPUT_FILE = '../data/frotz_builtin_walkthrough.csv'
    df, st_encoded, proj_down = pca_encoder(INPUT_FILE)
    make_plots(proj_down, st_encoded, df)    




    


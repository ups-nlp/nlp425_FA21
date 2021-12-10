

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt


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
    
    # The PCA centers but does not scale. Do we need to scale?
    pca = PCA(n_components=50)
    proj_down = pca.fit_transform(st_encoded)
    
    # Project back up to compare to original inputs
    proj_up = pca.inverse_transform(proj_down)
    
    return df, st_encoded, proj_down, pca, proj_up


def plot_inputs(st_encoded):
    """Plot the inputs to see what they look like"""

    ninputs = np.shape(st_encoded)[0]
    norig_features = np.shape(st_encoded)[1]
    
    plt.figure()
    plt.plot(np.arange(ninputs), st_encoded)
    plt.xlabel('observation')
    plt.ylabel('value')
    plt.title('Original: each curve represents the values for a particular feature')
    plt.show()
        
    plt.figure()
    plt.plot(np.arange(norig_features), st_encoded.T)
    plt.xlabel('feature')
    plt.ylabel('value')
    plt.title('Original: each curve represents the values for a particular observation')
        
def plot_components(proj_down):
    """Plot the principal components """
    
    ninputs = np.shape(proj_down)[0]
    ncomponents = np.shape(proj_down)[1]

    plt.figure()
    plt.subplot(311)
    plt.title('First 9 Principal Components')
    plt.plot(np.arange(ninputs), proj_down[:,:3])
    plt.ylabel('PCs 1-3')
    plt.subplot(312)
    plt.plot(np.arange(ninputs), proj_down[:,3:6])
    plt.ylabel('PCs 4-6')
    plt.subplot(313)
    plt.plot(np.arange(ninputs), proj_down[:,6:9])
    plt.ylabel('PCs 7-9')
    plt.xlabel('observation')
        
    plt.figure()
    plt.plot(np.arange(ncomponents), proj_down.T, '.-')
    plt.xlabel('component')
    plt.ylabel('value')
    plt.title('Encoded: each curve represents the values for a particular observation')
    
def plot_variance_explained(pca):
    """Plot the variance explained by the principal components"""
    # Plot the explained variance
    
    tot_var_explained = [np.sum(pca.explained_variance_ratio_[:i]) \
                         for i in range(len(pca.explained_variance_ratio_))]
    explained_str = str(np.round(tot_var_explained[-1]*100))
    
    plt.figure()
    plt.subplot(211)
    plt.plot(pca.explained_variance_ratio_,'o-')
    plt.ylabel('Variance explained (fraction)')
    plt.title('The first 50 components explain ' + explained_str + '% of the ' \
              + 'variance')
    
    plt.subplot(212)
    plt.plot(tot_var_explained, 'o-')
    plt.xlabel('principal component')
    plt.ylabel('Total variance explained (fraction)')
    
def plot_reconstructed(st_encoded, proj_up, iinput, ifeature):
    """
    Compare original and reconstructed for a particular input (all features)
    and a particular feature (all inputs)
    """
    plt.figure()
    plt.subplot(211)
    plt.plot(st_encoded[:,ifeature], label = 'original')
    plt.plot(proj_up[:,ifeature], label = 'reconstructed')
    plt.ylabel('value')
    plt.legend()
    plt.title('Values for feature ' + str(ifeature))
    
    plt.subplot(212)
    plt.plot(st_encoded[:,ifeature] - proj_up[:,ifeature])
    plt.xlabel('observation')
    plt.ylabel('original - reconstructed')
    plt.show()
        
    plt.figure()
    plt.subplot(211)
    plt.plot(st_encoded[iinput,:], label = 'original')
    plt.plot(proj_up[iinput,:], label = 'reconstructed')
    plt.ylabel('value')
    plt.legend()
    plt.title('Values for observation ' + str(iinput))

    plt.subplot(212)
    plt.plot(st_encoded[iinput,:] - proj_up[iinput,:])
    plt.xlabel('feature')
    plt.ylabel('original - reconstructed')



if __name__ == "__main__":
    INPUT_FILE = '../data/frotz_builtin_walkthrough.csv'
    
    # Get the data and encode it
    df, st_encoded, proj_down, pca, proj_up = pca_encoder(INPUT_FILE)
    
    # Make figures
    #plot_inputs(st_encoded)
    #plot_components(proj_down)
    #plot_variance_explained(pca)
    plot_reconstructed(st_encoded, proj_up, 0, 0)






    


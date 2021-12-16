

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import pandas as pd


class PCAencoder:

    def __init__(self, observations, components=50):
        """ Initialize by training the model using the observations """
        self.obs = observations
        # Encode observations as numerical vectors
        self.st_encoded = self.st_encode(self.obs)
        
        # The PCA model (includes centering but not scaling)
        self.pca = PCA(n_components = components)
        self.proj_down = self.pca.fit_transform(self.st_encoded)
        
        # Project back up to compare to original inputs
        self.proj_up = self.pca.inverse_transform(self.proj_down)
        
        
    def st_encode(self, obs):
        """Run the encoder on input data"""
        # Encode the observations
        # The inputs are size: inputs x features: (396 x 768)
        # The inputs vary from -1.2 to 1.4, so they are already pretty well
        # centered and scaled, and they are already flat
        # This encoder produces vectors of length 768: multi-qa-mpnet-base-dot-v1
        # This one produces vectors of length 3xx: all-MiniLM-L6-v2
        sentTrans = SentenceTransformer('multi-qa-mpnet-base-dot-v1')
        return np.array([sentTrans.encode([x])[0] for x in obs])
    
    def encode(self, obs):
        """
        Encode the inputs using PCA
        @param obs Observations, as text strings
        """
        # Convert the observations to numerical vectors
        st_encoded = self.st_encode(obs)
        # Perform the PCA transform to reduce the number of features
        return self.pca.transform(st_encoded)
        
    
    def plot_input(self, index):
        """Plot the observation at the given index
        @param index The index to the observation to plot"""
        
        norig_features = np.shape(self.st_encoded)[1]
                    
        plt.figure()
        plt.plot(np.arange(norig_features), self.st_encoded[index,:])
        plt.xlabel('feature')
        plt.ylabel('value')
        plt.title('Original: each curve represents the values for a particular observation')    

    def plot_inputs(self):
        """Plot the inputs to see what they look like"""
    
        st_encoded = self.st_encoded
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
            
    def plot_components(self):
        """Plot the principal components """
        
        proj_down = self.proj_down
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
        
    def plot_variance_explained(self):
        """Plot the variance explained by the principal components"""
        pca = self.pca
        
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

    def plot_reconstructed(self, iinput, ifeature):
        """
        Compare original and reconstructed for a particular input (all features)
        and a particular feature (all inputs)
        """
        st_encoded = self.st_encoded
        proj_up = self.proj_up
        
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
        
    def save_to_file(self, fname):
        """Write the obs to a CSV file"""
        np.savetxt(fname, self.proj_down, delimiter = ',')
        
        
        
if __name__ == "__main__":
    """Create an instance of the PCAencoder"""

    # Read in the data from the walkthrough    
    INPUT_FILE = 'data/frotz_builtin_walkthrough.csv'
    df = pd.read_csv(INPUT_FILE)
    observations = (df['Observation']).values
    actions = (df['Action']).values              # The labels, or correct answers

    # Create the PCA encoder
    pcaEncoder = PCAencoder(observations)
    
    # Check the shapes of things
    print('There are', len(observations), 'observations')
          
    rows = np.shape(pcaEncoder.proj_down)[0]
    cols = np.shape(pcaEncoder.proj_down)[1]
    print('The encoded observations are:', str(rows), 'x', str(cols))
    print('E.g.', str(rows) , 'observations each of length', str(cols))
    print('There are', len(actions), 'actions')

    # Save the encoded inputs to a file
    #pcaEncoder.save_to_file('data/encoded_observations.csv')
    
    # Plot the encoded observation 1 (2nd obs)
    pcaEncoder.plot_input(1)
        
  
        
    

    


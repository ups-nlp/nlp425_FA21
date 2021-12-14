import random
from jericho import FrotzEnv
from sentence_transformers import SentenceTransformer
import numpy as np
import re
from operator import truediv
from operator import add

class word_Transformer:
    """Interface for an word transfromer"""

    def create_vect(observation:str) -> list:
        """Takes in a string and transforms into a vector"""
        raise NotImplementedError
    
class glove(word_Transformer):
    """
    uses glove to transfrom words into vectors
    """
    def __init__(self):
        """
        Initialize the class by setting instance variables for enemies,
        weapons, and movements
        """
        self.vocab_vectors, self.word2id = self.embed_vocab()

    def create_vect(self, observation:str) -> list:
        """
        Takes an observation and returns a 50 dimensional vector representation of it

        @param list: a list containing the vectors in the vocab which are represented by lists
        @param dict: a dictionary linking a word to a vector
        @param str: a string containing an observation

        @return list: A list representing a 50 dimensional normalized vector
        """
        obs_split = observation.split(' ')
        num_words = 0

        #Creates an empty list of size 50 to be filled in
        vect_size = 50
        avg_vect = [0] * vect_size

        for word in obs_split:
            #Check if the word is in our vocab, if it is add it to the vector
            if(self.word2id.get(word) is not None):
                id = self.word2id.get(word)
                norm_vect = self.vocab_vectors[id]
                avg_vect = list(map(add, avg_vect, norm_vect))
                num_words +=1
            #else:
                #print("Word not in the vocab: " + word)

        words = [num_words] * vect_size
        avg_vect = list(map(truediv, avg_vect, words))

        return(avg_vect)

    def embed_vocab() -> (list, dict):
        """
        Reads in the vocab and vector GloVemaster files in from the data folder.
        Returns a dictionary matching a word to an index and a list of the vectors

        @return list: A normalized list of vectors, one for each word
        @return dict: A dictionary with a word as a key and the id for the list as the value
        """
        with open("./data/vocab.txt", 'r') as f:
            #Read in the list of words
            words = [word.rstrip().split(' ')[0] for word in f.readlines()]

        with open("./data/vectors.txt", 'r') as f:
            #word --> [vector]
            vectors = {}
            for line in f:
                vals = line.rstrip().split(' ')
                word = vals[0]
                vec = vals[1:]
                vectors[word] = [float(x) for x in vec]

        #Compute size of vocabulary
        vocab_size = len(words)
        word2id = {w: idx for idx, w in enumerate(words)}
        id2word = {idx: w for idx, w in enumerate(words)}

        vector_dim = len(vectors[id2word[0]])
        #print("Vocab size: " + str(vocab_size))
        #print("Vector dimension: " + str(vector_dim))

        #Create a numpy matrix to hold word vectors
        W = np.zeros((vocab_size, vector_dim))
        for word, v in vectors.items():
            if word == '<unk>':
                continue
            W[word2id[word], :] = v

        #Normalize each word vector to unit length
        W_norm = np.zeros(W.shape)
        d = (np.sum(W ** 2, 1) ** (0.5))
        W_norm = (W.T / d).T

        return W_norm, word2id

class sen_transformer(word_Transformer):
    """
    uses Sentance transfromers to transfrom words to vectors 
    """
    def __init__(self):
        """
        Initialize the class by up sentance transformer
        """
        # set model for sentence transformers
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def create_vect(self, observation:str) -> list:
        # get the oberservation

        # Encode the observation
        query_vec = self.model.encode([observation])[0]
        """Takes in the history and returns the next action to take"""
        raise query_vec
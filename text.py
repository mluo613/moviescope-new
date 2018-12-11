from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import os, numpy as np
import pickle
import importlib
import sys

class Text:

    def __init__(self):
        self.count_vect = CountVectorizer()
        self.tfidf_transformer = TfidfTransformer()

    def fit_transform(self, plots):
        x = self.count_vect.fit_transform(plots)
        x_train = self.tfidf_transformer.fit_transform(x).toarray()
        return x_train

    def transform(self, plots):

        if type(plots)==list:
            x = self.count_vect.transform(plots)
            x = self.tfidf_transformer.fit_transform(x)
            x = x.toarray()
#            print "Warning: Data is returned as Scipy sparse Matrix. To use it as numpy matrices, please convert the returned  matrix using '.toarray()' method."
        elif type(plots)==str:
            x = self.count_vect.transform([plots])
            x = self.tfidf_transformer.fit_transform(x).toarray()[0]
        else:
            print ("TypeError: Please pass a string (plot) or a list of plots")
            return None

        return x


class WordEmbeddings:

    def __init__(self, maxSequenceLength = 3000, maxWords = 50000, embeddingDim = 300):

        self.embeddings_index = {}
        GLOVE_DIR='data/glove_embeddings/'
        f = open(os.path.join(GLOVE_DIR, 'glove.6B.{}d.txt'.format(embeddingDim)),'rb')
        print ('Loading GloVe Embedding for {} dimensions.'.format(embeddingDim))
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            self.embeddings_index[word] = coefs
        f.close()
        print ("Loaded.")
        self.maxWords = maxWords
        self.embeddingDim = embeddingDim
        self.maxSequenceLength = maxSequenceLength

    def fit_transform(self, plots):

        self.tokenizer = Tokenizer(self.maxWords)
        self.tokenizer.fit_on_texts(plots)
        #creating word embedding matrix
        word_index = self.tokenizer.word_index
        self.num_words = min(self.maxWords, len(word_index))
        self.embedding_matrix = np.zeros((self.num_words+1, self.embeddingDim))

        for word, index in word_index.items():
            if index > self.maxWords:
                continue
            embedding_vector = self.embeddings_index.get(word)
            if embedding_vector is not None:
                self.embedding_matrix[index] = embedding_vector
                

        sequences = self.tokenizer.texts_to_sequences(plots)
        x = pad_sequences(sequences, maxlen=self.maxSequenceLength)
        return x


    def transform(self, plots, reverse=False):

        sequences = self.tokenizer.texts_to_sequences(plots)
        x = pad_sequences(sequences, maxlen=self.maxSequenceLength)

        if reverse:
            sequences_reverse = [seq[::-1] for seq in sequences]
            x_reverse = pad_sequences(sequences_reverse, maxlen=self.maxSequenceLength)
            return x,x_reverse
        return x

    def get_embedding_matrix(self):
        return self.embedding_matrix

    def get_num_words(self):
        return self.num_words

    def get_embedding_dim(self):
        return self.embeddingDim

    def get_input_length(self):
        return self.maxSequenceLength


def test(obj):
    plots = ["The movie Avatar is about all blue and alien creatures in a planet named same as a radio station", "The\
    Dark Knight is an amazing movie which features a villian defeating the superhero and the movie ending with some\
    terrible choices made by the head commisioner of a great city.", "Captain America is nobody's favorite super hero. Oh my."]

    xTrain = obj.fit_transform(plots)
    print (obj.transform("How can I be sure of you in this city"))
    print (obj.transform(['The movie about super hero n this city', 'I know the villian is an amazing hero']))


def _create_model_alpha():

    """ Function to test the text-features passed to deep model """
    """ do not want to cram create_model file """
    return

def get_training_labels(filename):
    with (open("data/"+filename+".p", "rb")) as openfile:
        loaded = pickle.load(openfile)
        return loaded
 
def main():
    #obj = WordEmbeddings()
    #test(obj)
    with (open("fasttext_train.p", "rb")) as openfile:
        loaded = pickle.load(openfile, encoding='latin1')
        print( loaded)
if __name__=="__main__":
    main()

from nltk.classify.util import apply_features
from nltk import NaiveBayesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import os
import nltk
from flask import current_app
import pickle, re
import collections
from nltk.tokenize import  word_tokenize

class SpamClassifier:

    def load_model(self, model_name):
        model_file = os.path.join(current_app.config['ML_MODEL_UPLOAD_FOLDER'], model_name+'.pk')
        model_word_features_file = os.path.join(current_app.config['ML_MODEL_UPLOAD_FOLDER'],model_name +'_word_features.pk')
        with open(model_file, 'rb') as mfp:
            self.classifier = pickle.load(mfp)
        with open(model_word_features_file, 'rb') as mwfp:
            self.word_features = pickle.load(mwfp)


    def extract_tokens(self, text, target):
        """returns array of tuples where each tuple is defined by (tokenized_text, label)
         parameters:
                text: array of texts
                target: array of target labels

        NOTE: consider only those words which have all alphabets and atleast 3 characters.
        """
        text = pd.Series(text)
        target = pd.Series(target)
        train = pd.concat([text,target],axis=1)
        tr = [(r[0],r[1]) for index,r in train.iterrows()]
#         tr = tr[1:5]
        self.tr = tr
        test = [(nltk.word_tokenize(r[0]),r[1]) for r in tr]
        filtered_test = [(y.lower(),x[1]) for x in test for y in x[0] if y.isalpha() and len(y)>3]
        return filtered_test



    def get_features(self, corpus):
        """
        returns a Set of unique words in complete corpus.
        parameters:- corpus: tokenized corpus along with target labels

        Return Type is a set
        """
        words = [word[0] for word in corpus]
        return set(words)


    def extract_features(self, document):
        """
        maps each input text into feature vector
        parameters:- document: string

        Return type : A dictionary with keys being the train data set word features.
                      The values correspond to True or False
        """
        return {document: document in [t2 for t1 in self.trained_data for t2 in word_tokenize(t1)] }

    def train(self, text, labels):
        """
        Returns trained model and set of unique words in training data
        """
        self.word_features= self.get_features(self.extract_tokens(text, labels))
        self.trained_data=text
        
        t=[({word: word in (x[0]) for word  in self.word_features }  ,x[1])  for x in self.extract_tokens(text, labels) if x[0] is not None]
        self.classifier= NaiveBayesClassifier.train(t)
        
        return self.classifier, self.word_features
        

    def predict(self, text):
        """
        Returns prediction labels of given input text.
        """
        p=[]
        if isinstance(text,str):
            test_features = {word.lower(): (word in word_tokenize(text.lower())) for word in self.word_features}
            p.append(self.classifier.classify(test_features))
        if isinstance(text,list):
            for t in text:
                test_features = {word.lower(): (word in word_tokenize(t.lower())) for word in self.word_features}
                p.append(self.classifier.classify(test_features))
        
        
        return p



if __name__ == '__main__':
    print('Done')
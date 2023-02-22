import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import time
import sys
from pdb import set_trace
import numpy as np
from collections import Counter
from gensim.utils import tokenize
from gensim.parsing import preprocessing

##################################
sys.path.insert(0,'../..')
from my_evaluation import my_evaluation
from my_GA import my_GA


class my_model():

    def obj_func(self, predictions, actuals, pred_proba=None):
        # One objectives: higher f1 score
        eval = my_evaluation(predictions, actuals, pred_proba)
        return [eval.f1()]
    
    def binary_oversample(self, y, ratio = 1, replace = False):
        #ratio how many samples of smaller class per sample of bigger class
        class_counts = Counter(y)
        min_class = min(class_counts, key=class_counts.get)
        max_class = max(class_counts, key=class_counts.get)
        max_class_value = class_counts[max(class_counts, key=class_counts.get)]
        min_class_value = class_counts[min_class]
        desired_samples_oversampled = np.floor(max_class_value*ratio)
        number_of_times_duplicate = int(np.floor(desired_samples_oversampled/min_class_value))
        samples_remaining_required = int(desired_samples_oversampled % min_class_value)
        oversample_index = []
        oversample_index.extend(list(np.where(y == max_class)[0]))
        for i in range(number_of_times_duplicate):
            oversample_index.extend(list(np.where(y == min_class)[0]))
            
        random_sampling_index = list(np.random.choice(list(np.where(y == min_class)[0]), samples_remaining_required, replace = replace))
        oversample_index.extend(random_sampling_index)
        
        return oversample_index



    def fit(self, X, y):
        # do not exceed 29 mins

        ########################
        #might try this to make the 4 classes equal
        # oversample_index = self.binary_oversample(y, ratio = 1)
        # X_sample = X.loc[oversample_index, :]
        # y_sample = y[oversample_index].to_numpy()

        X_sample = X
        y_sample = y   

        tokenizer = lambda x: tokenize(x, lowercase=True) #gensim tokenizer

        #strip tags first, the punctiation, then non alphanum, then remove white space
        description = list(X_sample['Feature Request Text'])
        description = [preprocessing.strip_tags(d) for d in description]
        description = [preprocessing.strip_punctuation(d) for d in description]
        description = [preprocessing.strip_non_alphanum(d) for d in description]
        description = [preprocessing.strip_multiple_whitespaces(d) for d in description]



        sub_tf = True

        count_vect_description = CountVectorizer(tokenizer=tokenizer, stop_words=None, strip_accents = None)
        X_description_counts = count_vect_description.fit_transform(description)
        self.count_vect_description = count_vect_description

        tfidf_transformer_description = TfidfTransformer(sublinear_tf=sub_tf)
        X_description_tfidf = tfidf_transformer_description.fit_transform(X_description_counts)
        self.tfidf_transformer_description = tfidf_transformer_description
        X_description_tfidf.shape
        X_description = X_description_tfidf.todense()

        X_rest = np.zeros((len(y_sample), 1))
        XX = np.concatenate((X_description, X_rest), axis = 1)

        self.clf = SGDClassifier(loss = 'hinge', penalty = 'l2', alpha = 0.0001).fit(np.asarray(XX), y_sample)

        return

    def predict(self, X):
        X_sample = X


        #strip tags first, the punctiation, then non alphanum, then remove white space
        description = list(X_sample['Feature Request Text'])
        description = [preprocessing.strip_tags(d) for d in description]
        description = [preprocessing.strip_punctuation(d) for d in description]
        description = [preprocessing.strip_non_alphanum(d) for d in description]
        description = [preprocessing.strip_multiple_whitespaces(d) for d in description]

    


        X_description_counts = self.count_vect_description.transform(description)
        X_description_tfidf = self.tfidf_transformer_description.transform(X_description_counts)
        X_description_tfidf.shape
        X_description = X_description_tfidf.todense()



        X_rest = np.zeros((X_description.shape[0], 1))
        XX = np.concatenate((X_description, X_rest), axis = 1)

        predictions = self.clf.predict(np.asarray(XX))


        return predictions


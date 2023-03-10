import sys
import time
from collections import Counter
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier as SGD
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB as MNB
from sklearn.naive_bayes import ComplementNB as CNB
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from pdb import set_trace
from gensim.utils import tokenize
from gensim.parsing import preprocessing
from nltk import word_tokenize

from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import GridSearchCV

##################################
sys.path.insert(0, "../..")
from my_evaluation import my_evaluation
from my_GA import my_GA
from skopt import BayesSearchCV, space
from symspellpy.symspellpy import SymSpell
import pkg_resources
#consider miss spelling using symspell


class my_model:
    def obj_func(self, predictions, actuals, pred_proba=None):
        # One objectives: higher f1 score
        eval = my_evaluation(predictions, actuals, pred_proba)
        return [eval.f1()]

    def binary_oversample(self, y, ratio=1, replace=False):
        # ratio how many samples of smaller class per sample of bigger class
        class_counts = Counter(y)
        min_class = min(class_counts, key=class_counts.get)
        max_class = max(class_counts, key=class_counts.get)
        max_class_value = class_counts[max(class_counts, key=class_counts.get)]
        min_class_value = class_counts[min_class]
        desired_samples_oversampled = np.floor(max_class_value * ratio)
        number_of_times_duplicate = int(
            np.floor(desired_samples_oversampled / min_class_value)
        )
        samples_remaining_required = int(desired_samples_oversampled % min_class_value)
        oversample_index = []
        oversample_index.extend(list(np.where(y == max_class)[0]))
        for i in range(number_of_times_duplicate):
            oversample_index.extend(list(np.where(y == min_class)[0]))

        random_sampling_index = list(
            np.random.choice(
                list(np.where(y == min_class)[0]),
                samples_remaining_required,
                replace=replace,
            )
        )
        oversample_index.extend(random_sampling_index)

        return oversample_index
    
    def multi_class_oversample(self, y, ratio=1, replace=False):
        # ratio how many samples of smaller class per sample of bigger class

        class_counts = Counter(y)
        oversample_index = []
        for class_label in class_counts.keys():
            min_class = class_label
            max_class = max(class_counts, key=class_counts.get)
            max_class_value = class_counts[max(class_counts, key=class_counts.get)]
            min_class_value = class_counts[min_class]
            desired_samples_oversampled = np.floor(max_class_value * ratio)
            number_of_times_duplicate = int(
                np.floor(desired_samples_oversampled / min_class_value)
            )
            samples_remaining_required = int(desired_samples_oversampled % min_class_value)
            # oversample_index.extend(list(np.where(y == max_class)[0]))
            for i in range(number_of_times_duplicate):
                oversample_index.extend(list(np.where(y == min_class)[0]))

            random_sampling_index = list(
                np.random.choice(
                    list(np.where(y == min_class)[0]),
                    samples_remaining_required,
                    replace=replace,
                )
            )
            oversample_index.extend(random_sampling_index)

        return oversample_index

    def fit(self, X, y):
        # do not exceed 29 mins
        X.reset_index(drop=True, inplace=True)
        y.reset_index(drop=True, inplace=True)

        ########################
        # might try this to make the 4 classes equal with over sampling
        # oversample_index = self.multi_class_oversample(y, ratio = 1)
        # X_sample = X.loc[oversample_index, :].reset_index(drop=True)
        # y_sample = y[oversample_index].reset_index(drop=True).to_numpy()


        #If we don't want over sampling use these
        X_sample = X
        y_sample = y

        tokenizer = lambda x: tokenize(x, lowercase=True) #gensim tokenizer
        wnl = WordNetLemmatizer()

        # Set max_dictionary_edit_distance to avoid spelling correction
        sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
        sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

        tokenizer = lambda x: tokenize(x, lowercase=True)  # gensim tokenizer

        
        description = list(X_sample["Feature Request Text"])

        # strip tags first, then punctiation, then non alphanum, then remove white space
        description = [preprocessing.strip_tags(d) for d in description]
        description = [preprocessing.strip_numeric(d) for d in description]
        description = [preprocessing.strip_punctuation(d) for d in description]
        description = [preprocessing.remove_stopwords(d) for d in description]
        description = [preprocessing.strip_non_alphanum(d) for d in description]
        description = [preprocessing.strip_multiple_whitespaces(d) for d in description]
        description = [preprocessing.split_alphanum(d) for d in description]
        print('Performing spelling corrections. This is an intensive step')
        description = [sym_spell.word_segmentation(d).corrected_string if len(d) > 0 else d for d in description]
        print('Spelling corrections completed')
        description = [wnl.lemmatize(d) for d in description]
        description = [preprocessing.stem_text(d) for d in description]

        #These are the steps the paper did exactly
        # description = [preprocessing.strip_numeric(d) for d in description]
        # description = [preprocessing.strip_punctuation(d) for d in description]
        # description = [preprocessing.remove_stopwords(d) for d in description]
        # description = [wnl.lemmatize(d) for d in description]
        # description = [preprocessing.stem_text(d) for d in description]



        sub_tf = False

        count_vect_description = CountVectorizer(
            tokenizer=tokenizer, stop_words="english", strip_accents="unicode"
        )
        X_description_counts = count_vect_description.fit_transform(description)

        self.count_vect_description = count_vect_description

        tfidf_transformer_description = TfidfTransformer(sublinear_tf=sub_tf)
        X_description_tfidf = tfidf_transformer_description.fit_transform(
            X_description_counts
        )
        self.tfidf_transformer_description = tfidf_transformer_description
        X_description_tfidf.shape
        X_description = X_description_tfidf.todense()

        X_rest = np.zeros((len(y_sample), 1))

        # This is if we WANT to normalize for length of description
        XX = np.concatenate((X_description, X_rest), axis=1)

        # This is if we don't want to normalize for length of description
        # XX = np.concatenate((X_description_counts.todense(), X_rest), axis = 1)

        # self.clf = SGDClassifier(loss = 'hinge', penalty = 'l2', alpha = 0.0001).fit(np.asarray(XX), y_sample)

        # for grid search
        # param_grid = [{'C': [0.5,1,10,100],
        #        'gamma': ['scale', 1,.1,.01,.001,.0001],
        #        'kernel':['rbf']}]

        # param_grid = {'C': space.Real(1,100),
        #        'gamma': space.Real(.0001,1)}

        # print('running Baysian Search')
        # optimal_params = BayesSearchCV(SVC(class_weight = 'balanced'), param_grid, cv=2, scoring='f1_macro', verbose=0)
        # optimal_params.fit(np.asarray(XX), y_sample)
        # print(optimal_params.best_params_)

        # self.clf = SVC(C=5.435,gamma=0.2884).fit(np.asarray(XX), y_sample)
        # self.clf = CNB().fit(np.asarray(XX), y_sample)
        # self.clf = MNB().fit(np.asarray(XX), y_sample)
        # self.clf = RF().fit(np.asarray(XX), y_sample)
        # self.clf = DT().fit(np.asarray(XX), y_sample)
        # self.clf = SGD().fit(np.asarray(XX), y_sample)
        self.clf = MLP().fit(np.asarray(XX), y_sample)


        
        return

    def predict(self, X):
        X_sample = X


        wnl = WordNetLemmatizer()

        # Set max_dictionary_edit_distance to avoid spelling correction
        sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
        sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)


        
        description = list(X_sample["Feature Request Text"])

        # strip tags first, the punctiation, then non alphanum, then remove white space
        description = [preprocessing.strip_tags(d) for d in description]
        description = [preprocessing.strip_numeric(d) for d in description]
        description = [preprocessing.strip_punctuation(d) for d in description]
        description = [preprocessing.remove_stopwords(d) for d in description]
        description = [preprocessing.strip_non_alphanum(d) for d in description]
        description = [preprocessing.strip_multiple_whitespaces(d) for d in description]
        description = [preprocessing.split_alphanum(d) for d in description]
        print('Performing spelling corrections for predictions. This is an intensive step')
        description = [sym_spell.word_segmentation(d).corrected_string if len(d) > 0 else d for d in description]
        print('Spelling corrections completed')
        description = [wnl.lemmatize(d) for d in description]
        description = [preprocessing.stem_text(d) for d in description]

        #These are the steps the paper did exactly
        # description = [preprocessing.strip_numeric(d) for d in description]
        # description = [preprocessing.strip_punctuation(d) for d in description]
        # description = [preprocessing.remove_stopwords(d) for d in description]
        # description = [wnl.lemmatize(d) for d in description]
        # description = [preprocessing.stem_text(d) for d in description]


        # This is if we WANT to normalize for length of description
        X_description_counts = self.count_vect_description.transform(description)
        X_description_tfidf = self.tfidf_transformer_description.transform(
            X_description_counts
        )
        X_description_tfidf.shape
        X_description = X_description_tfidf.todense()

        # This is if we DON'T WANT to normalize for length of description
        # X_description_counts = self.count_vect_description.transform(description)
        # X_description = X_description_counts.todense()

        X_rest = np.zeros((X_description.shape[0], 1))
        XX = np.concatenate((X_description, X_rest), axis=1)

        predictions = self.clf.predict(np.asarray(XX))

        return predictions

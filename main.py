import time
import numpy as np
import pandas as pd
import pickle
import nltk
from sklearn import tree
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from nltk.tokenize import word_tokenize
#nltk.download("stopwords")
from sklearn.neural_network import MLPClassifier

from modelFactory import *
from dataSanitizer import *
from dataAugmentor import *
from testTrainSplit import *
from modelContainer import *

#augmenting imports:
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as nafc
from nlpaug.util import Action
#nltk.download('averaged_perceptron_tagger')
import os
os.environ["MODEL_DIR"] = '../model/'

# Loading data from File
df = pd.read_csv("data2.csv")

# Cleaning Data
data = DataSanitizer().clean(df)

#legacy, not used
#label_value_data = data[['label', 'label_value']].drop_duplicates().sort_values('label')

n_splits = 100
# ORIGINAL Split the data in to testing and training
x_train_notUsed, x_test, y_train_notUsed, y_test = TestTrainSplit.splitData(data, n_splits)

#TODO rename newDF this to data
aug = AugmentFactory().SYNONYM_REPLACEMENT()
newDF = DataAugmentor().augment(data, 30, aug) #int = the amount of data to be augmented
#newDF = DataAugmentor().augment(newDF, 33, AugmentFactory().RANDOM_WORD_DELETION())
#newDF = DataAugmentor().augment3(newDF, 60)

#Use this if not augmenting:
#newDF = data

# ORIGINAL Split the data in to testing and training
x_train, x_test_notUsed, y_train, y_test_notUsed = TestTrainSplit.splitData(newDF, n_splits)

# bestAccuracy = -1
# bestModel = None
for algorithm in Algorithm:
    #print(algorithm)
    print(algorithm.name)
    #print(algorithm.value)
    for i in range(0, n_splits):
        container = ModelContainer(algorithm)
        trainedModel = container.train(x_train[i], y_train[i], newDF)

        #grid search
        # parameters = ParamFactory.create(algorithm)
        # #parameters = {'max_features': ['auto', 'log2']}
        # #clf = GridSearchCV(trainedModel, parameters)
        # clf = GridSearchCV(trainedModel,  param_grid=parameters, verbose=1, n_jobs=-1)
        # params = clf.get_params().keys()
        # clf.fit(container.textVect_x_Train, y_train[i])
        # for key in clf.cv_results_:
        #     print(key)
        #     print(clf.cv_results_[key])

        accuracy = container.evaluate(trainedModel, x_test[i], y_test[i], newDF)

        # if (accuracy > bestAccuracy):
        #     bestAccuracy = accuracy
        #     bestModel = trainedModel

        # TODO python thing to make file at pickle location
        # and then this function underneath to dump the pickle
        #pickle.dump(trainedModel, open('pickle/' + algorithm.name + str(i) + '.pickle', 'wb'))
#pickle.dump(bestModel, open('randomForest.pickle', 'wb'))


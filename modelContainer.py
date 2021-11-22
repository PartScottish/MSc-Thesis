from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import confusion_matrix

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

from modelFactory import *
from dataSanitizer import *
from dataAugmentor import *
from testTrainSplit import *


# TODO rename this - this method is for building and testing the model
class ModelContainer:
    def __init__(self, algorithm):
        self.textVect_x_Train = None
        self.textVect = TfidfVectorizer(max_features=500, stop_words=stop_words)
        self.algorithm = algorithm

    def train(self, x_train, y_train, newDF):
        newDF = newDF.dropna(subset=['voice', 'sentence'])
        self.textVect_x_Train = self.textVect.fit_transform(x_train).toarray()
        # Training the model
        model = ModelFactory.create(self.algorithm)
        model.fit(self.textVect_x_Train, y_train)
        return model

    def evaluate(self, model, x_test, y_test, data):
        #TODO check a .accuracy()
        # Test on everything
        x_Trans = self.textVect.transform(x_test).toarray()
        accuracy = model.score(x_Trans, y_test)
        y_pred = model.predict(x_Trans)


    #Below are printout statements for results
        textVect_x_Test = self.textVect.fit_transform(x_test).toarray()
        print("accuracy_score: %.3f" % accuracy)
        #print(model.predict(textVect_x_Test))

        #TODO F score is printing accuracy (incorrect)
        y_true = y_test.array.to_numpy()
        f1s = f1_score(y_true, y_pred, average='micro')
        #print(f1s)

        #print(confusion_matrix(y_test, y_pred))


        index = data.index
        # number_of_rows = len(index)
        # print(number_of_rows)
        #print(data.sample(10))
        # print(x_Trans[:10])
        # print(textVect_x_Train[:10])

        return accuracy
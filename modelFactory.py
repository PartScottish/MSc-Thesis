from enum import Enum, auto
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

#TODO Note that any params listed below and/or in the gridsearch method are unlikely to be the final ones used
# in the training and are at various stages of testing.

#TODO Consider renaming algorithm names to match
class Algorithm(Enum):
    RANDOM_FOREST = auto()
    #MNB = auto()
    #BNB = auto()
    #SGD = auto()
    #XB = auto()
    #NN = auto()

    #GNB = auto() #not used, bad science

class ModelFactory():
    def create(algorithm):
        return {
            Algorithm.RANDOM_FOREST: RandomForestClassifier(random_state=0, class_weight='balanced', criterion='gini'),
            #Algorithm.MNB: MultinomialNB(),
            #Algorithm.BNB: BernoulliNB(),
            #Algorithm.SGD: SGDClassifier(max_iter=1000, tol=1e-3, penalty='l1', alpha=0.0001, n_iter_no_change=25),
            #Algorithm.XB: GradientBoostingClassifier(n_estimators=500, learning_rate=0.01, subsample=0.8, max_features='auto')
            #Algorithm.NN:  MLPClassifier(early_stopping=True, max_iter=1000, solver='adam', alpha=0.0001, hidden_layer_sizes=(25, 400))
            #Algorithm.GNB: GaussianNB(), #Do not use, < 0r algorithm
        }[algorithm]


#Used in SGD Grid Search Params
# penalty = ['l1', 'l2']
# alpha = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
# class_weight = 'balanced'
# solver = ['liblinear', 'saga']
#

#Used in MLPC Grid Search Params

# hidden_layer_sizes = [(50,50,50), (50,100,50), (100,)]
# activation = ['tanh', 'relu']
# solver = ['sgd', 'adam', 'lbfgs']
# alpha = [0.0001, 0.05]
# learning_rate = ['constant','adaptive']

# class ParamFactory():
#     def create(algorithm):
#         return {
#             #Algorithm.RANDOM_FOREST: {'max_features': ['auto', 'log2']},
#             #Algorithm.SGD: dict(penalty=penalty, alpha=alpha)
#             Algorithm.NN: dict(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver, alpha=alpha, learning_rate=learning_rate)
#
#
#         }[algorithm]
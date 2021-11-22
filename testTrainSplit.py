from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold

class TestTrainSplit:
    def splitData(data, n_splits):
        #x_train, x_test, y_train, y_test = train_test_split(data['sentence'], data['label'], random_state=1)
        x_train, x_test, y_train, y_test = [],[],[],[]

        # Split data with Stratified K-Fold
        skf = StratifiedKFold(n_splits=n_splits)
        t = data.label
        for train_index, test_index in skf.split(data, t):
            thing1 = data.iloc[train_index]
            thing2 = data.iloc[test_index]
            x_train.append(thing1.sentence)
            x_test.append(thing2.sentence)
            y_train.append(thing1.label)
            y_test.append(thing2.label)

        #return [x_train], [x_test], [y_train], [y_test]
        return x_train, x_test, y_train, y_test
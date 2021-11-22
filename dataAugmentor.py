import pandas as pd
from augmentFactory import *
#augmenting imports:
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as nafc
from nlpaug.util import Action
#nltk.download('averaged_perceptron_tagger')
import os
os.environ["MODEL_DIR"] = '../model/'

class DataAugmentor:
    def augment(self, data, percentage, aug):
        # Augmenting the data
        newDF = []
        for index, row in self.getPercentData(data, percentage).iterrows():  # change int to be percentage of data augmented
            text = row["sentence"]
            augmented_text = aug.augment(text)
            newRow = row
            newRow["sentence"] = augmented_text
            newDF.append(newRow)

        newDF = pd.DataFrame(newDF)
        newDF = pd.concat([data, newDF], ignore_index=True, sort=False)
        return newDF

    #TODO Make more science-y
    def getPercentData(self, df, percentage):
        moduloValue = 100 // percentage
        newDF = []
        count = 0
        for index, row in df.iterrows():
            if count % moduloValue == 0:
                newDF.append(row)
            count += 1
        return pd.DataFrame(newDF)


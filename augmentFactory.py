from enum import Enum, auto
import pandas as pd
#augmenting imports:
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as nafc
from nlpaug.util import Action
#nltk.download('averaged_perceptron_tagger')

import os
os.environ["MODEL_DIR"] = 'model/'

class AugmentFactory():
    def __init__(self):
        self._synRep = None
        self._randLet = None
        self._nonRandLet = None
        self._randWordInsert = None
        self._randWordSub = None
        self._randWordDel = None
        # self._backTrans = None #Removed, not useful with non-normalised sentence structure from spoken transcript.

    def RANDOM_LETTER_REPLACEMENT(self):
        if (self._randLet == None):
            self._randLet = nac.RandomCharAug(action="substitute")
        return self._randLet

    def RANDOM_WORD_INSERTION(self):
        if (self._randWordInsert == None):
            self._randWordInsert = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action="insert")
        return self._randWordInsert

    def RANDOM_WORD_DELETION(self):
        if (self._randWordDel == None):
            self._randWordDel = naw.RandomWordAug()
        return self._randWordDel

    def SYNONYM_REPLACEMENT(self):
        if (self._synRep == None):
            self._synRep = naw.SynonymAug(aug_src='ppdb', model_path=os.environ.get("MODEL_DIR") + 'ppdb-2.0-tldr')
        return self._synRep

    def RANDOM_WORD_SUBSTITUTION(self):
        if (self._randWordSub == None):
            self._randWordSub = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action="substitute")
        return self._randWordSub

# Not used/needed in final test-loop
    # def NON_RANDOM_LETTER_REPLACEMENT(self):
    #     if (self._nonRandLet == None):
    #         self._nonRandLet = nac.OcrAug()
    #     return self._nonRandLet

# #SUPER slow, few GB download.
#     def BACK_TRANSLATION(self):
#         if (self._backTrans == None):
#             self._backTrans = naw.BackTranslationAug(
#                 # from_model_name='transformer.wmt19.en-de',
#                 # to_model_name='transformer.wmt19.de-en')
#                 from_model_name = 'facebook/wmt19-en-de',
#                 to_model_name = 'facebook/wmt19-de-en')
#         return self._backTrans

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
#nltk.download("stopwords")

class DataSanitizer:
    def clean(self, data):
        df = data[["API", "label", "turn id", "sentence"]]
        df = df.rename(columns={"API": "label", "label": "score", "turn id": "voice"})

        df["voice"] = df["voice"].apply(lambda voice: self.renameVoice(voice))

        df["sentence"] = df["sentence"].apply(lambda sentence: self.tidySentence(sentence))

        # dropping all NA's from voice and any errors caused in sentences after cleaning
        df = df.dropna(subset=['voice', 'sentence'])
    # This turnes voice to 0 or 1 for integer-based/requiring analysis
        # df["voice_num"] = df["voice"].apply(lambda voice : voiceToNum(voice))

        col = ['label', 'score', 'voice', 'sentence']
        df = df[col]
        df['label_value'] = df['label'].factorize()[0]
        return df

    # function to rename all the voices in the turns to simply T for therapist or P for patient
    def renameVoice(self, voice):
        if type(voice) == str and (voice[0] == "T" or voice[0] == "P"):  # 1 is T and 0 is P
            return voice[0]

    def voiceToNum(self, voice):
        if (voice) == "T":  # 1 is T and 0 is P
            return 0.0
        if (voice) == "P":
            return 1.0
        return None

    # function for cleaning sentences to remove noise for predictions
    def tidySentence(self, sentence):
        # Removing T/P noise
        sentence = sentence.split(':')
        if len(sentence) > 1:
            sentence = " ".join(sentence[1:])
        else:
            sentence = " ".join(sentence)

        # Removing \t noise
        sentence = sentence.split("\t")
        if len(sentence) > 1:
            sentence = " ".join(sentence[1:])
        else:
            sentence = " ".join(sentence)

        return sentence




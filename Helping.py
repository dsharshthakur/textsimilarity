# Step 1: importing necessary libraries
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

nltk.download("punkt")

# Step 2: read the text data set
dataset = pd.read_csv("DataNeuron_Text_Similarity.csv")

# Step 3: creating corpus to train the word2vec
training_text = dataset["text1"].str.cat(sep=" ") + dataset["text2"].str.cat(sep=" ")
training_text = " ".join(filter(lambda item: item.isalnum(), training_text.split(" ")))

# Step 2: sentence tokenization - required by Word2Vec
tokenized_sentence = sent_tokenize(training_text.lower())


# Step 5: word tokenization - required by Word2Vec

tokens = [sent.split(" ") for sent in tokenized_sentence]

# Step 6: creating model object
model = Word2Vec(tokens, window=5, min_count=1)


# Step 7 : creating TextSimilarity class that will be responsible for getting the final output
class TextSimilarity:
    # constructor
    def __init__(self, text_1, text_2, model):
        self.txt_a = text_1
        self.txt_b = text_2
        self.model = model

    # performing work embedding
    def vectorization(self):

        # word tokenization of input texts received
        tokens_a = [wrd.lower() for wrd in self.txt_a.split(" ") if wrd.isalpha()]
        tokens_b = [wrd.lower() for wrd in self.txt_b.split(" ") if wrd.isalpha()]

        # storing sentence vector
        text_1_vector = []
        text_2_vector = []

        # text 1 - embedding
        for wrd in tokens_a:
            try:
                # fetching the word vector from the model
                # considering the mean of the vectors - as words are expressed using a vector
                vector = np.mean(self.model.wv[wrd])
            except:
                # if word not found in the model
                vector = 0

            text_1_vector.append(vector)

        # text 2 - embedding
        for wrd in tokens_b:
            try:
                vector = np.mean(self.model.wv[wrd])
            except:
                vector = 0

            text_2_vector.append(vector)

        # length of the longest string
        max_len = max(len(text_1_vector), len(text_2_vector))

        # performing post - padding if required
        if len(text_1_vector) == max_len:
            text_2_vector = text_2_vector + [0] * (max_len - len(text_2_vector))
        else:
            text_1_vector = text_1_vector + [0] * (max_len - len(text_1_vector))

        return text_1_vector, text_2_vector

    def calculate(self):
        # calculating the cosine similarity

        vec_1, vec_2 = TextSimilarity.vectorization(self)
        score = cosine_similarity([vec_1], [vec_2])
        score = score.ravel()[0]
        return score


texta = "he is very angry"
textb = "his mood is good as he scored good"

score = TextSimilarity(texta, textb, model)
print(score.calculate())

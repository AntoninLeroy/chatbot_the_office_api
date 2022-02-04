from scipy import sparse
import pandas as pd
import re
import numpy as np
from numpy.random import choice
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

class similaritySentenceDetector():
    
    def pre_processing(self, text):
    
        # patern definition

        remove_didascalies = re.compile(r"\[.*?\]")
        add_space_before_questionmark = re.compile(r"([?!])")

        # processing

        text = re.sub(remove_didascalies, "", text)
        text = re.sub(add_space_before_questionmark, " \\1", text)
        text = text.lower() #lowercase
        text = text.replace('[”#$%&()*+-/:;<=>@[\]/^_`{|}~…]','') #remove special char
        text = text.replace('’',"'") #correct symbol
        text = text.replace(',',"") #correct symbol
        clean_text = " ".join(text.split()) # remove unecessary spaces

        return clean_text

    def get_cosine_similarity(self, X, vectorizer, input_sentence, top_n):
        
        # reference : https://stackoverflow.com/questions/12118720/python-tf-idf-cosine-to-find-document-similarity
        
        clean_input_sentence = self.pre_processing(input_sentence)
        cosine_similarities = linear_kernel(vectorizer.transform([input_sentence]), X).flatten()
        related_docs_indices = cosine_similarities.argsort()[:-1 -top_n:-1]
        cosine_similarities_values = cosine_similarities[related_docs_indices]

        return related_docs_indices, cosine_similarities_values
    
    def get_response(self, input_sentence, df, X, vectorizer, character_to_awnser):
        
        #pre-process input sentence
        input_sentence = similaritySentenceDetector.pre_processing(self, input_sentence)
        
        character_to_awnser = [item.lower() for item in character_to_awnser]
        
        # does the input contain names of specific characters ?
        match = list(set(character_to_awnser) & set(input_sentence.lower().split()))
        
        if len(match) > 0:
            to_awnser = match
        else:
            to_awnser = character_to_awnser
        
        related_docs_indices, cosine_similarities_values = similaritySentenceDetector.get_cosine_similarity(self, X, vectorizer, input_sentence, 500)
        
        awnsers = []
        idx_awnser = []
        score_awnser = []
        
        # loop through most similar sentences
        for idx, score in zip(related_docs_indices, cosine_similarities_values):
            
            # not out of index condition
            if idx < len(df) - 1:
               
                # if the repply is one of the allowed character
                if df.Character[idx + 1].lower() in to_awnser:
                    
                    awnsers.append(df.Line[idx + 1])
                    idx_awnser.append(idx)
                    score_awnser.append(score)
        
        prob = score_awnser / sum(score_awnser)
        try:
            randomNumberList = choice(awnsers, 1, p=prob)
        except:
            randomNumberList = choice(awnsers, 1)
        rand = awnsers.index(randomNumberList[0])

        character = df.Character[int(idx_awnser[rand]) + 1] # we simply select the next sentence in the script as response
        response = df.Line[int(idx_awnser[rand]) + 1]
        similarity = round(score_awnser[rand],2)

        return character, response, similarity
import numpy as np
import nltk
import spacy
# nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def tokenize(sentence):
    """
    split sentence into array of words/tokens
    a token can be a word or punctuation character, or number
    """
    return nltk.word_tokenize(sentence)


def stem(word):
    """
    stemming = find the root form of the word
    examples:
    words = ["organize", "organizes", "organizing"]
    words = [stem(w) for w in words]
    -> ["organ", "organ", "organ"]
    """
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, words):
    """
    return bag of words array:
    1 for each known word that exists in the sentence, 0 otherwise
    example:
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bog   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
    """
    # stem each word
    sentence_words = [stem(word) for word in tokenized_sentence]
    # initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1

    return bag

def semantic_sim(u,v):
    return (u @ v) / (np.sqrt(sum(u**2)) * np.sqrt(sum(v**2)))

def sentence_similarity(sent1, sent2):
    nlp = spacy.load("en_core_web_sm")
    
    # tokenization
    sent1_list = nlp(sent1) 
    sent2_list = nlp(sent2)
    # print('lists: ', sent1_list, sent2_list)
    
    # remove stop words from the string
    sent1_set = {w.lemma_.lower() for w in sent1_list if not w.is_punct and not w.is_stop and not w.text.isdigit()} 
    sent2_set = {w.lemma_.lower() for w in sent2_list if not w.is_punct and not w.is_stop and not w.text.isdigit()} 
    # print('sets: ', sent1_set, sent2_set)
    
    sent_sims = []

    # form a set containing keywords of both strings  
    for w1 in sent1_set:
        word_sim = 0
        w1 = nlp(w1)

        for w2 in sent2_set:
            w2 = nlp(w2)
            # print('words: ', w1, w2)
            u = w1.vector
            v = w2.vector
            ss = semantic_sim(u, v)
            # print('ss: ', ss)
            # print(ss > word_sim)
            if ss > word_sim:
                word_sim = ss
        sent_sims.append(word_sim)
    
    return(sum(sent_sims)/len(sent_sims))

def select_answer(user_sent, responses: list):
    similarities = []
    for response in responses:
        similarities.append(sentence_similarity(user_sent, response))
    return similarities.index(max(similarities))
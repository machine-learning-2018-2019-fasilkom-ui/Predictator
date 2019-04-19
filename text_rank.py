import numpy as np
import pycrfsuite

from nltk.tag import CRFTagger
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


ct = CRFTagger()
ct.set_model_file('dataset/all_indo_man_tag_corpus_model.crf.tagger')

factory = StemmerFactory()
stemmer = factory.create_stemmer()



# Text rank function
def load_stopwords():
    stop_words = []
    with open('stopwords-id.txt') as f:
        for word in f.readlines():
            stop_words.append(word.strip('\n'))
    return stop_words

stop_words = load_stopwords()

def build_vocabulary(sentences):
    word_to_ix = {}
    ix_to_word = {}

    for sent in sentences:
        for word in sent:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
                ix_to_word[len(ix_to_word)] = word
    
    return word_to_ix, ix_to_word

def stem_sentence(sentence):
    return [stemmer.stem(word) for word in sentence]

def filter_words(sentence):
    flatten = lambda l: [item for sublist in l for item in sublist]
    tags = set(["NN", "NNS", "NNP", "JJ", "JJR", "JJS"])
    filtered_sentence = []
    for word, tag in flatten(ct.tag_sents([sentence])):
        if tag not in tags:
            continue
        
        if word.lower() in stop_words:
            continue
        
        filtered_sentence.append(word)
    
    return filtered_sentence

def normalize_sentence(sentence, lowercase=True):
    if lowercase:
        return [word.lower() for word in sentence]
    
def filter_sentences(sentences, lowercase=True, stem=True):
    norm_sents = [normalize_sentence(s, lowercase) for s in sentences]
    
    filtered_sents = [filter_words(sent) for sent in norm_sents]
    filtered_sents = list(filter(None, filtered_sents))

    if stem:
        return [stem_sentence(sent) for sent in filtered_sents]
    
    return filtered_sents

def build_coo_matrix(sentences, word_to_ix):
    S = np.zeros((len(word_to_ix), len(word_to_ix)))

    for sent in sentences:
        for src, target in zip(sent[:-1], sent[1:]):
            if src.lower() == target.lower():
                continue
            
            S[word_to_ix[src]][word_to_ix[target]] = 1
            S[word_to_ix[target]][word_to_ix[src]] = 1
    
    return normalize_matrix(S)

def normalize_matrix(S):
    for i in range(len(S)):
        if S[i].sum() == 0:
            S[i] = np.ones(len(S))
        
        S[i] /= S[i].sum()
    
    return S

def pagerank(A, eps=0.0001, d=0.85):
    R = np.ones(len(A))
    
    while True:
        r = np.ones(len(A)) * (1 - d) + d * A.T.dot(R)
        if abs(r - R).sum() <= eps:
            return r
        R = r
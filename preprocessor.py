from data import open_dataset
from collections import defaultdict
from copy import deepcopy
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from spacy.lemmatizer import Lemmatizer
from spacy.lang.id import LOOKUP
from nltk.tag import CRFTagger
import numpy as np
import re
import json
import pycrfsuite

ANALYSIS_DATA_DIR = "analysis/"
STOP_MIN_THRESHOLD = 10
STOP_MAX_THRESHOLD_PERCENTAGE = 0.999

def idf_counter(data):
    document_frequency = defaultdict(int)
    idf = defaultdict(float)
    flatten = lambda l: [item for sublist in l for item in sublist]
    n_doc = len(data)
    for i in data:
        paragraphs = flatten(i["paragraphs"])
        words = set(flatten(paragraphs))
        for word in words:
            document_frequency[word] += 1
    for k,v in document_frequency.items():
        idf[k] = np.log2(1.0*n_doc/v)
    return idf

def weighted_tf(data):
    tf = []
    flatten = lambda l: [item for sublist in l for item in sublist]
    for i in data:
        paragraphs = flatten(i["paragraphs"])
        words = flatten(paragraphs)
        doc_tf = defaultdict(float)
        term_count = len(words)
        for word in words:
            doc_tf[word] += 1
        for k,v in doc_tf.items():
            doc_tf[k] = 1.0*v/term_count
        tf.append(doc_tf)
    return tf

def tf_idf(data):
    flatten = lambda l: [item for sublist in l for item in sublist]
    tf = weighted_tf(data)
    idf = idf_counter(data)
    for idx, doc in enumerate(data):
        doc["tf_idf_score"] = []
        for i,paragraph in enumerate(doc["paragraphs"]):
            doc["tf_idf_score"].append([])
            doc["tf_idf_score"][i] = []
            for j,sentence in enumerate(paragraph):
                doc["tf_idf_score"][i].append(0.0)
                for word in sentence:
                    doc["tf_idf_score"][i][j] += tf[i][word]*idf[word]
            # Normalization
        doc_max_tf_idf = max(flatten(doc["tf_idf_score"]))
        for i,paragraph in enumerate(doc["paragraphs"]):
            for j,sentence in enumerate(paragraph):
                doc["tf_idf_score"][i][j] /= doc_max_tf_idf
        print(doc["tf_idf_score"])
        break
    return data

def term_frequency_counter(data):
    term_frequency = defaultdict(int)
    flatten = lambda l: [item for sublist in l for item in sublist]
    for i in data:
        paragraphs = flatten(i["paragraphs"])
        words = flatten(paragraphs)
        for word in words:
            term_frequency[word] += 1
    unique_token_count = len(term_frequency.keys())
    with open(ANALYSIS_DATA_DIR+"word_frequencies.json", 'w') as f:
        for k,v in sorted(term_frequency.items(), key=lambda item: (item[1], item[0])):
            f.write(json.dumps({k:v}))
            f.write("\n")
    return term_frequency

def tokenizer():
    # This step is skipped beacause the data set already tokenize
    pass

def stopword_remover(data):
    stop_words = []
    vocabulary_frequency = term_frequency_counter(data)
    n_unique_token = len(vocabulary_frequency)
    max_threshold = STOP_MAX_THRESHOLD_PERCENTAGE * n_unique_token
    sorted_word = sorted(vocabulary_frequency.items(), key=lambda item: (item[1], item[0]))
    for idx, (k,v) in enumerate(sorted_word):
        # Dibawah threshold
        if v <= STOP_MIN_THRESHOLD:
            stop_words.append(k)
        if idx >= int(max_threshold):
            stop_words.append(k)
    stop_words = set(stop_words)
    print(len(stop_words))
    for doc in data:
        doc["stopped_paragraphs"] = []
        for i,paragraph in enumerate(doc["paragraphs"]):
            doc["stopped_paragraphs"].append([])
            doc["stopped_paragraphs"][i] = []
            for k,sentence in enumerate(paragraph):
                doc["stopped_paragraphs"][i].append([word for word in sentence
                                                if word not in stop_words])
    return data
    # pass

def word_stemmer(data):
    # Skip lemmatizer if this step is chosen
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    for doc in data:
        doc["stemmed_paragraphs"] = []
        for i,paragraph in enumerate(doc["stopped_paragraphs"]):
            doc["stemmed_paragraphs"].append([])
            doc["stemmed_paragraphs"][i] = []
            for k,sentence in enumerate(paragraph):
                doc["stemmed_paragraphs"][i].append([])
                for idx, word in enumerate(sentence):
                    if re.match(r'^[a-z]*$', word) or idx==0:
                        # only stem lower cased alphabet word and beginning of a sentence
                        doc["stemmed_paragraphs"][i][k].append(stemmer.stem(word))
                    else:
                        doc["stemmed_paragraphs"][i][k].append(word)
    return data

def word_lemmatizer(data):
    lemmatizer = Lemmatizer(lookup=LOOKUP)
    for doc in data:
        doc["lemma_paragraphs"] = []
        for i,paragraph in enumerate(doc["stopped_paragraphs"]):
            doc["lemma_paragraphs"].append([])
            doc["lemma_paragraphs"][i] = []
            for k,sentence in enumerate(paragraph):
                doc["lemma_paragraphs"][i].append([])
                for idx, word in enumerate(sentence):
                    doc["lemma_paragraphs"][i][k].append(lemmatizer(word, u"NOUN")[0])
    return data

def F1(s,S):
    f=[]
    flattened=[val for sublist in S["paragraphs"] for val in sublist]
    temp=deepcopy(flattened)
    temp.remove(s)
    flattened2=[val for sublist in temp for val in sublist]
    d1=list(set(s).intersection(flattened2))
    for i in range(len(flattened)-1):
        a=flattened[i]
        temp=deepcopy(flattened)
        temp.pop(i)
        flattened2=[val for sublist in temp for val in sublist]
        d2=list(set(a).intersection(flattened2))
        f.append(len(d2))
    return len(d1)/max(f)

def F6(s,S):
    res=0
    title=(S["id"].split("-",1)[1]).split("-")
    for x in (s):
        if x in (title):
            res=res+1
    res=res/len(title)
    return res

def F7(s,S):
    res=0
    if s==S["paragraphs"][0][0] or s==S["paragraphs"][0][len(S["paragraphs"][0])-1] or s==S["paragraphs"][len(S["paragraphs"])-1][0] or s==S["paragraphs"][len(S["paragraphs"])-1][len(S["paragraphs"][len(S["paragraphs"])-1])-1]:
        res=1
    return res

def F1_extraction(data):
    flatten = lambda l: [item for sublist in l for item in sublist]
    for doc in data:
        doc["F1"]=[]
        for paragraph in doc["paragraphs"]:
            list_f1=[]
            for sentence in paragraph:
                list_f1.append(F1(sentence,doc))
            doc["F1"].append(list_f1)
        doc["F1"]=flatten(doc["F1"])
    return data

def F6_extraction(data):
    flatten = lambda l: [item for sublist in l for item in sublist]
    for doc in data:
        doc["F6"]=[]
        for paragraph in doc["paragraphs"]:
            list_f6=[]
            for sentence in paragraph:
                list_f6.append(F6(sentence,doc))
            doc["F6"].append(list_f6)
        doc["F6"]=flatten(doc["F6"])
    return data

def F7_extraction(data):
    flatten = lambda l: [item for sublist in l for item in sublist]
    for doc in data:
        doc["F7"]=[]
        for paragraph in doc["paragraphs"]:
            list_f7=[]
            for sentence in paragraph:
                list_f7.append(F7(sentence,doc))
            doc["F7"].append(list_f7)
        doc["F7"]=flatten(doc["F7"])
    return data

def pos_tagger(data):
    flatten = lambda l: [item for sublist in l for item in sublist]
    ct = CRFTagger()
    ct.set_model_file('dataset/all_indo_man_tag_corpus_model.crf.tagger')
    for category in data:
        category['word_tag'] = []
        for paragraph in category['paragraphs']:
            list_tag_kalimat = []
            for kalimat in paragraph:
                tag_kalimat = ct.tag_sents([kalimat])
                list_tag_kalimat.append(tag_kalimat)
            category['word_tag'].append(list_tag_kalimat)
        category['word_tag'] = flatten(category['word_tag'])
    return data

def demo():
    flatten = lambda l: [item for sublist in l for item in sublist]
    data = [open_dataset("dev", 1),open_dataset("train", 1), open_dataset("test", 1)]
    data = flatten(data)
    data = tf_idf(data)
    # print("Tag every word in paragraphs...")
    # data = pos_tagger(data)
    print("jumlah dokumen %i"%len(data))
    print(data[1]["paragraphs"])
    print("---------")
    print("Removing StopWord")
    data = stopword_remover(data)
    print(data[1]["stopped_paragraphs"])
    print("---------")
    print("Stemming")
    data = word_stemmer(data)
    print(data[1]["stemmed_paragraphs"])
    print("---------")
    print("Lemmatization")
    data = word_lemmatizer(data)
    print(data[1]["lemma_paragraphs"])


if __name__ == "__main__":
    demo()

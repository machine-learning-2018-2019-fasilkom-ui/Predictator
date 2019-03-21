from data import open_dataset
from collections import defaultdict
from copy import deepcopy
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from spacy.lemmatizer import Lemmatizer
from spacy.lang.id import LOOKUP
import re
import json

ANALYSIS_DATA_DIR = "analysis/"
STOP_MIN_THRESHOLD = 10
STOP_MAX_THRESHOLD_PERCENTAGE = 0.999

def word_frequency_counter(data):
    word_frequency = defaultdict(int)
    flatten = lambda l: [item for sublist in l for item in sublist]
    for i in data:
        paragraphs = flatten(i["paragraphs"])
        words = flatten(paragraphs)
        for word in words:
            word_frequency[word] += 1
    unique_token_count = len(word_frequency.keys())
    with open(ANALYSIS_DATA_DIR+"word_frequencies.json", 'w') as f:
        for k,v in sorted(word_frequency.items(), key=lambda item: (item[1], item[0])):
            f.write(json.dumps({k:v}))
            f.write("\n")
    return word_frequency

def tokenizer():
    # This step is skipped beacause the data set already tokenize
    pass

def stopword_remover(data):
    stop_words = []
    vocabulary_frequency = word_frequency_counter(data)
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

def pos_tagger():
    # May be..?
    pass

def demo():
    flatten = lambda l: [item for sublist in l for item in sublist]
    data = [open_dataset("dev", 1),open_dataset("train", 1), open_dataset("test", 1)]
    data = flatten(data)
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

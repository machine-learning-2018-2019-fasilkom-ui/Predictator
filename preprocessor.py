from data import open_dataset
from collections import defaultdict
import json

ANALYSIS_DATA_DIR = "analysis/"
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
    # pass

def stemmer():
    # Skip lemmatizer if this step is chosen
    pass

def lemmatizer():
    # Skip stemmer if this step is chosen
    pass

def pos_tagger():
    # May be..?
    pass

def demo():
    flatten = lambda l: [item for sublist in l for item in sublist]
    data = [open_dataset("dev", 1),open_dataset("train", 1), open_dataset("test", 1)]
    data = flatten(data)
    print("jumlah dokumen %i"%len(data))
    data_st = stopword_remover(data)

if __name__ == "__main__":
    demo()

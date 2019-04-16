from data import open_dataset
from preprocessor import pos_tagger, stopword_remover, word_stemmer, word_lemmatizer
from feature_extraction import compute_feature
from model.lead3 import Lead3

def main():
    flatten = lambda l: [item for sublist in l for item in sublist]
    data = [open_dataset("dev", 1),open_dataset("train", 1), open_dataset("test", 1)]
    data = flatten(data)
    for doc in data:
        doc["predicted_label"] = {}
        doc["predicted_label"]["lead3"] = Lead3.predict(doc)
    print(data[0])

if __name__ == "__main__":
    main()

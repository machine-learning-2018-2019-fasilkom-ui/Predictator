import json

DATASET_DIR = "dataset/indosum/"
FILENAME_FORMAT = {"dev":"dev.%.2i.jsonl", "train":"train.%.2i.jsonl",
                "test":"test.%.2i.jsonl"}

def open_dataset(type, fold):
    print(type, FILENAME_FORMAT[type]%fold)
    with open(DATASET_DIR+FILENAME_FORMAT[type]%fold, 'r') as datafile:
        data = []
        line = datafile.readline()
        while line:
            data.append(json.loads(line))
            line = datafile.readline()
    return data

def demo():
    dataset_type = list(FILENAME_FORMAT.keys())
    fold = range(1,6)
    for dataset in dataset_type:
        for k in fold:
            print("Opening dataset %s fold %i"%(dataset, k))
            data = open_dataset(dataset, k)

if __name__ == "__main__":
    demo()

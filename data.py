import json

DATASET_DIR = "dataset/indosum/"
FILENAME_FORMAT = {"dev":"dev.%.2i.jsonl", "train":"train.%.2i.jsonl",
                "test":"test.%.2i.jsonl"}

def default_url_parser(url):
    return url.lower().rstrip("/").split("/")[-1].split("?")[0].rstrip("-").split("-")

def merdeka_url_parser(url):
    return url.split("/")[-1].split(".")[0].split("-")

def suara_url_parser(url):
    return "-".join(default_url_parser(url)).split("#")[0].split("-")

def juaranet_parser(url):
    url_parsed = url.split("/")
    domain = url_parsed[2]
    try:
        title = url_parsed[6].split("-")[1:]
    except:
        title = url_parsed[-1].split("-")[1:]
    if domain == "www.juara.net":
        title = "-".join(title).rstrip(".").split(".")
    return title

def goal_indonesia_url_parser(url):
    url_parsed = url.split("/")
    domain = url_parsed[2]
    lang_code = url_parsed[4 if domain=="m.goal.com" else 3]
    news_category = url_parsed[5 if domain=="m.goal.com" else 4]
    title = []
    if lang_code == "id-ID":
        if news_category == "match":
            title = url_parsed[5].split("-")
        else:
            title = url_parsed[-1].split("?")[0].split("-")
    elif lang_code == "id":
        title = url_parsed[5].split("-")
    return title

def get_title(key, url):
    return {
        "merdeka": merdeka_url_parser(url),
        "suara": suara_url_parser(url),
        "juara.net": juaranet_parser(url),
        "goal indonesia": goal_indonesia_url_parser(url)
    }.get(key, default_url_parser(url))

def open_dataset(type, fold):
    print(type, FILENAME_FORMAT[type]%fold)
    with open(DATASET_DIR+FILENAME_FORMAT[type]%fold, 'r') as datafile:
        data = []
        line = datafile.readline()
        while line:
            data.append(json.loads(line))
            line = datafile.readline()
    return data

def write_dataset(data, type, fold):
    print(type, FILENAME_FORMAT[type]%fold)
    with open(DATASET_DIR+FILENAME_FORMAT[type]%fold, 'w') as datafile:
        for datum in data:
            datafile.write(json.dumps(datum))
            datafile.write("\n")

def quick_stat(data):
    # Len berita
    # total kalimat
    # total vocab
    #
    n_data = len(data)
    flatten = lambda l: [item for sublist in l for item in sublist]
    vocab = []
    sentence_counter = 0
    pos_label_counter = 0
    neg_label_counter = 0
    summary_length = 0
    for doc in data:
        for sentence in flatten(doc["paragraphs"]):
            sentence_counter += 1
            vocab.append(sentence)
        for label in flatten(doc["gold_labels"]):
            if label == True:
                pos_label_counter += 1
            else:
                neg_label_counter += 1
        summary_length += len(doc["summary"])

    vocab = flatten(vocab)
    print("#Berita          : %d"%(n_data))
    print("#Kalimat         : %d"%(sentence_counter))
    print("#Kalimat/#Berita : %f"%(1.0*sentence_counter/n_data))
    print("#Kata/#Berita    : %f"%(1.0*len(vocab)/n_data))
    print("#Unique Token    : %d"%(len(set(vocab))))
    print("#Summary/#Berita : %f"%(1.0*summary_length/n_data))
    print("#Positive Label  : %d"%(pos_label_counter))
    print("#Negative Label  : %d"%(neg_label_counter))
    print("Prevalence Data  : %f"%(1.0*pos_label_counter/sentence_counter))

def fix_dataset():
    dataset_type = list(FILENAME_FORMAT.keys())
    fold = range(1,6)
    flatten = lambda l: [item for sublist in l for item in sublist]
    data_dict = dict()
    for k in fold:
        data = []
        for dataset in dataset_type:
            print("Opening dataset %s fold %i"%(dataset, k))
            data.append(open_dataset(dataset, k))
        data = flatten(data)
        for doc in data:
            if doc["id"] in data_dict:
                if len(flatten(doc["gold_labels"])) > len(flatten(data_dict[doc["id"]]["gold_labels"])):
                    data_dict[doc["id"]] = doc
            else:
                data_dict[doc["id"]] = doc
    print("rewriting dataset")
    for k in fold:
        for dataset in dataset_type:
            data = open_dataset(dataset, k)
            for idx in range(len(data)):
                data[idx] = data_dict[data[idx]["id"]]
            write_dataset(data, dataset, k)

def demo():
    dataset_type = list(FILENAME_FORMAT.keys())
    fold = range(1,6)
    for dataset in dataset_type:
        for k in fold:
            print("Opening dataset %s fold %i"%(dataset, k))
            data = open_dataset(dataset, k)
    fix_dataset()
    # source_key = []
    # for datum in data:
    #     source_key.append(datum["source"])
    #     if datum["source"] == "juara.net":
    #         print(datum["source_url"])
    #         print(get_title(datum["source"], datum["source_url"]))
    #         print("------------------")
    # print(set(source_key))
    flatten = lambda l: [item for sublist in l for item in sublist]
    data = [open_dataset("dev", 1),open_dataset("train", 1), open_dataset("test", 1)]
    data = flatten(data)
    quick_stat(data)

if __name__ == "__main__":
    demo()

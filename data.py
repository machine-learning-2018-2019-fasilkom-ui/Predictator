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

def demo():
    dataset_type = list(FILENAME_FORMAT.keys())
    fold = range(1,6)
    for dataset in dataset_type:
        for k in fold:
            print("Opening dataset %s fold %i"%(dataset, k))
            data = open_dataset(dataset, 1)

    source_key = []
    for datum in data:
        source_key.append(datum["source"])
        if datum["source"] == "juara.net":
            print(datum["source_url"])
            print(get_title(datum["source"], datum["source_url"]))
            print("------------------")
    print(set(source_key))

if __name__ == "__main__":
    demo()

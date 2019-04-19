from data import open_dataset, get_title
from preprocessor import stopword_remover, word_stemmer, word_lemmatizer, pos_tagger, pre_processed_all
from collections import defaultdict
from copy import deepcopy
from text_rank import filter_sentences, build_vocabulary, build_coo_matrix, pagerank

import re
import json
import numpy as np

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

def isf_counter(data):
    all_isf = []
    flatten = lambda l: [item for sublist in l for item in sublist]
    for idx,i in enumerate(data):
        # all_isf.append([])
        sentences = flatten(i["paragraphs"])
        n_sentence = len(sentences)
        sentence_frequency = defaultdict(int)
        isf = defaultdict(float)
        for sentence in sentences:
            # print(sentence)
            for word in set(sentence):
                sentence_frequency[word] += 1
            # print(sentence_frequency)
        for k,v in sentence_frequency.items():
            isf[k] = np.log2(1.0*n_sentence/v)
        all_isf.append(isf)
        # print(all_isf)
        # break
    return all_isf

def weighted_doc_tf(data):
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

def f1(s,S):
    flattened=[val for sublist in S["paragraphs"] for val in sublist]
    temp=deepcopy(flattened)
    temp.remove(s)
    flattened2=[val for sublist in temp for val in sublist]
    d1=list(set(s).intersection(flattened2))
    return len(d1)

def f2(s,S):
    global f
    temp=deepcopy(S)
    temp["paragraphs"].pop(idx)
    flattened=[val for sublist in temp["paragraphs"] for val in sublist]
    flattened2=[val for sublist in flattened for val in sublist]
    d1=list(set(s).intersection(flattened2))
    temp=deepcopy(S)
    flattened=[val for sublist in temp["paragraphs"] for val in sublist]
    if flag==1:
        f=[]
        for j in range(len(S["paragraphs"])):
            for i in range(len(S["paragraphs"][j])):
                a=flattened[i]
                temp=deepcopy(S["paragraphs"])
                temp.pop(j)
                flattened2=[val for sublist in [val for sublist in temp for val in sublist] for val in sublist]
                d2=list(set(a).intersection(flattened2))
                f.append(len(d2))
        if max(f)==0:
            return 0
        else:
            return len(d1)/max(f)
    else:
        if max(f)==0:
            return 0
        else:
            return len(d1)/max(f)

def f3(s,S):
    tot=0
    for word in s:
        if word.isupper() or any(list(x.isupper() for x in word)):
            tot+=1
    return tot/len(s)

def f6(s,S):
    res = 0
    title = get_title(S["source"], S["source_url"])
    for x in (s):
        if x.lower() in (title):
            res = res+1
    res = 1.0*res/len(title)
    return res

def f7(s,S):
    res = 0
    if s==S["paragraphs"][0][0] or s==S["paragraphs"][0][len(S["paragraphs"][0])-1] or s==S["paragraphs"][len(S["paragraphs"])-1][0] or s==S["paragraphs"][len(S["paragraphs"])-1][len(S["paragraphs"][len(S["paragraphs"])-1])-1]:
        res = 1
    return res

def f1_extraction(data):
    # similarity sentence
    global flag
    flatten = lambda l: [item for sublist in l for item in sublist]
    for doc in data:
        doc["F1"]=[]
        flag=1
        for paragraph in doc["paragraphs"]:
            list_f1=[]
            for sentence in paragraph:
                res=f1(sentence,doc)
                if flag==1:
                    f=[]
                    flattened=[val for sublist in doc["paragraphs"] for val in sublist]
                    for i in range(len(flattened)):
                        a=flattened[i]
                        temp=deepcopy(flattened)
                        temp.pop(i)
                        flattened2=[val for sublist in temp for val in sublist]
                        d2=list(set(a).intersection(flattened2))
                        f.append(len(d2))
                    denom=max(f)
                    result=res/denom
                else:
                    result=res/denom
                list_f1.append(result)
                flag=0
            doc["F1"].append(list_f1)
        doc["F1"]=flatten(doc["F1"])
    return data

def f2_extraction(data):
    global idx,flag
    flatten=lambda l: [item for sublist in l for item in sublist]
    for doc in data:
        doc["F2"]=[]
        flag=1
        for idx,paragraph in enumerate(doc["paragraphs"]):
            list_f2=[]
            for sentence in paragraph:
                list_f2.append(f2(sentence,doc))
                flag=0
            doc["F2"].append(list_f2)
        doc["F2"]=flatten(doc["F2"])
    return data

def f3_extraction(data):
    # Unique Formatting
    for doc in data:
        doc["F3"]=[]
        for paragraph in doc["paragraphs"]:
            list_f3=[]
            for sentence in paragraph:
                list_f3.append(f3(sentence,doc))
            doc["F3"].append(list_f3)
        doc["F3"]=flatten(doc["F3"])
    return data

def f4_extraction(data):
    # Important cue phrases
    # NOT EXTRACTED DUE TO LACK OF CUE PHRASES DATA
    pass

def f5_extraction(data):
    # TF-IDF
    flatten = lambda l: [item for sublist in l for item in sublist]
    tf = weighted_doc_tf(data)
    idf = idf_counter(data)
    for idx, doc in enumerate(data):
        doc["F5"] = []
        for i,paragraph in enumerate(doc["paragraphs"]):
            doc["F5"].append([])
            doc["F5"][i] = []
            for j,sentence in enumerate(paragraph):
                doc["F5"][i].append(0.0)
                for word in sentence:
                    doc["F5"][i][j] += tf[i][word]*idf[word]
        doc_max_tf_idf = max(flatten(doc["F5"]))
        for i,paragraph in enumerate(doc["paragraphs"]):
            for j,sentence in enumerate(paragraph):
                doc["F5"][i][j] /= doc_max_tf_idf
    return data

def f6_extraction(data):
    # unigram overlap sentencce with title
    flatten = lambda l: [item for sublist in l for item in sublist]
    for doc in data:
        doc["F6"]=[]
        for paragraph in doc["paragraphs"]:
            list_f6=[]
            for sentence in paragraph:
                list_f6.append(f6(sentence,doc))
            doc["F6"].append(list_f6)
        doc["F6"]=flatten(doc["F6"])
    return data

def f7_extraction(data):
    # Paragraph location
    flatten = lambda l: [item for sublist in l for item in sublist]
    for doc in data:
        doc["F7"]=[]
        for paragraph in doc["paragraphs"]:
            list_f7=[]
            for sentence in paragraph:
                list_f7.append(f7(sentence,doc))
            doc["F7"].append(list_f7)
        doc["F7"]=flatten(doc["F7"])
    return data

def f8_extraction(data):
    # Trivial cue phrases
    # NOT EXTRACTED DUE TO LACK OF PHRASES INFORMATION
    pass

def f9_extraction(data):
    #Score for sentences contains Proper Noun
    # Must run pos_tagger() first
    for category in data:
        category['F9_score'] = []
        for paragraph in category['word_tag']:
            list_score_kalimat = []
            for kalimat in paragraph:
                tag_NNP = 0
                for tag in kalimat:
                    if tag[1]=='NNP':
                        tag_NNP += 1
                    else:
                        pass
                score = float(tag_NNP/len(kalimat))
                list_score_kalimat.append(score)
            category['F9_score'].append(list_score_kalimat)
    return data

def f10_extraction(data):
    # TF-ISF
    flatten = lambda l: [item for sublist in l for item in sublist]
    tf = weighted_doc_tf(data)
    isf = isf_counter(data)
    for idx, doc in enumerate(data):
        doc["F10"] = []
        for i,paragraph in enumerate(doc["paragraphs"]):
            doc["F10"].append([])
            doc["F10"][i] = []
            for j,sentence in enumerate(paragraph):
                doc["F10"][i].append(0.0)
                for word in sentence:
                    doc["F10"][i][j] += tf[i][word]*isf[idx][word]
            # Normalization
        doc_max_tf_isf = max(flatten(doc["F10"]))

        for i,paragraph in enumerate(doc["paragraphs"]):
            for j,sentence in enumerate(paragraph):
                doc["F10"][i][j] /= doc_max_tf_isf
    return data

# Text Rank get score
def f11_extraction(data):
    for category in data:
        category['Textrank_score'] = []
        for paragraph in category['paragraphs']:
            list_score_textrank = []
            for kalimat in paragraph:
                filtered_sentences = filter_sentences([kalimat])
                word_to_ix, ix_to_word = build_vocabulary(filtered_sentences)
                S = build_coo_matrix(filtered_sentences, word_to_ix)
                ranks = pagerank(S)
                score = ranks.sum()
                list_score_textrank.append(score)
            category['Textrank_score'].append(list_score_textrank)
    return data


def sentence_sentrality(data):
    #Score sentence based on overlap words with other sentences
    flatten = lambda l: [item for sublist in l for item in sublist]
    for category in data:
        category['Overlap_score'] = []
        for paragraph in category['paragraphs']:
            list_score_overlap = []
            for kalimat in paragraph:
                kalimat_lain = list(category['paragraphs'])
                kalimat_lain = flatten(kalimat_lain)
                kalimat_lain.remove(kalimat)
                kalimat_lain = flatten(kalimat_lain)
                overlap = len(set(kalimat)) + len(set(kalimat_lain)) - len(set(kalimat + kalimat_lain))
                overlap_score = float(overlap/len(set(kalimat + kalimat_lain)))
                list_score_overlap.append(overlap_score)
            category['Overlap_score'].append(list_score_overlap)
    return data

def compute_feature(data):
    data = f1_extraction(data)
    data = f2_extraction(data)
    # data = f3_extraction(data)
    data = f5_extraction(data)
    data = f6_extraction(data)
    data = f7_extraction(data)
    data = f9_extraction(data)
    data = f10_extraction(data)
    data = f11_extraction(data)
    return data

def save_feature(data, precomputed=False, file_dir="analysis/feature_set.jsonl"):
    data = data if precomputed else compute_feature(data)
    selected_field = ["id", "F1", "F2", "F3", "F5", "F6", "F7", "F9", "F10", "F11"]
    with open(file_dir, "w") as f:
        for datum in data:
            selected_data = {}
            for field in selected_field:
                if field in datum:
                    selected_data[field] = datum[field]
                else:
                    selected_data[field] = []
            f.write(json.dumps(selected_data))
            f.write("\n")

def demo():
    flatten = lambda l: [item for sublist in l for item in sublist]
    data = [open_dataset("dev", 1),open_dataset("train", 1), open_dataset("test", 1)]
    data = flatten(data)
    data = pre_processed_all(data)
    print("F1")
    data = f1_extraction(data)
    print(data[0])
    print("F2")
    data = f2_extraction(data)
    print(data[0])
    # data = f3_extraction(data)
    print("F5")
    data = f5_extraction(data)
    print("F6")
    data = f6_extraction(data)
    print("F7")
    data = f7_extraction(data)
    print("F9")
    data = f9_extraction(data)
    print("F10")
    data = f10_extraction(data)
    print("F11")
    data = f11_extraction(data)
    print(data[0])
    save_feature(data, precomputed=True)

if __name__ == "__main__":
    demo()

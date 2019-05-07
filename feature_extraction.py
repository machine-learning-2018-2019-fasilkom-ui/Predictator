from data import open_dataset, get_title
from preprocessor import stopword_remover, word_stemmer, word_lemmatizer, pos_tagger, pre_processed_all
from collections import defaultdict
from copy import deepcopy
from text_rank import filter_sentences, build_vocabulary, build_coo_matrix, pagerank

import re
import json
import numpy as np
import time
from datetime import timedelta

flatten = lambda l: [item for sublist in l for item in sublist]

def idf_counter(data):
    document_frequency = defaultdict(int)
    idf = defaultdict(float)
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
    for doc in data:
        doc["F1"]=[]
        for paragraph in doc["paragraphs"]:
            for sentence in paragraph:
                res=f1(sentence,doc)
                doc["F1"].append(res)
        max_f1 = max(doc["F1"])
        doc["F1"] = [f1_val/max_f1 for f1_val in doc["F1"]]
    return data

def f2_extraction(data):
    flatten=lambda l: [item for sublist in l for item in sublist]
    cout=0 #del
    for doc in data:
        doc["F2"]=[]
        flag=1
        for idx,paragraph in enumerate(doc["paragraphs"]):
            list_f2=[]
            for sentence in paragraph:
                temp=deepcopy(doc)
                temp["paragraphs"].pop(idx)
                flattened=[val for sublist in temp["paragraphs"] for val in sublist]
                flattened2=[val for sublist in flattened for val in sublist]
                res=len(list(set(sentence).intersection(flattened2)))
                temp=deepcopy(doc)
                flattened=[val for sublist in temp["paragraphs"] for val in sublist]
                if flag==1:
                    f=[]
                    for j in range(len(doc["paragraphs"])):
                        for i in range(len(doc["paragraphs"][j])):
                            a=flattened[i]
                            temp=deepcopy(doc["paragraphs"])
                            temp.pop(j)
                            flattened2=[val for sublist in [val for sublist in temp for val in sublist] for val in sublist]
                            d2=list(set(a).intersection(flattened2))
                            f.append(len(d2))
                    denom=max(f)
                    if denom==0:
                        result=0
                    else:
                        result=res/denom
                else:
                    if denom==0:
                        result=0
                    else:
                        result=res/denom
                list_f2.append(result)
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
        doc["F5"] = flatten(doc["F5"])
    return data

def f6_extraction(data):
    # unigram overlap sentencce with title
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
    for doc in data:
        doc["F7"]=[]
        for paragraph in doc["paragraphs"]:
            list_f7=[]
            n_sentence = len(paragraph)
            for idx, sentence in enumerate(paragraph):
                list_f7.append(1 if(idx==0 or idx==n_sentence-1) else 0)
            doc["F7"].append(list_f7)
        doc["F7"]=flatten(doc["F7"])
    return data

def f8_extraction(data):
    # Trivial cue phrases
    # NOT EXTRACTED DUE TO LACK OF PHRASES INFORMATION
    pass

def f9_extraction(data):
    #Sscorecore for sentences contains Proper Noun
    # Must run pos_tagger() first
    for category in data:
        category['F9'] = []
        for paragraph in category['word_tag']:
            list_score_kalimat = []
            for kalimat in paragraph:
                tag_NNP = 0
                for tag in kalimat:
                    if tag[1]=='NNP':
                        tag_NNP += 1
                    else:
                        continue
                score = float(tag_NNP/len(kalimat))
                list_score_kalimat.append(score)
            category['F9'].append(list_score_kalimat)
        category["F9"] = flatten(category["F9"])
    return data

def f10_extraction(data):
    # TF-ISF
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
        doc["F10"] = flatten(doc["F10"])
    return data

# Text Rank get score

def f11_extraction(data):
    for category in data:
        category['F11'] = []
        temp = []
        for paragraph in category['paragraphs']:
            list_score_textrank = []
            for kalimat in paragraph:
                filtered_sentences = filter_sentences([kalimat])
                word_to_ix, ix_to_word = build_vocabulary(filtered_sentences)
                S = build_coo_matrix(filtered_sentences, word_to_ix)
                ranks = pagerank(S)
                score = ranks.sum()
                list_score_textrank.append(score)
            temp.append(list_score_textrank)
        flatted_score = flatten(temp)
        max_score = max(flatted_score)
        for paragraph_score in temp:
            list_score_textrank = []
            for sentence_score in paragraph_score:
                list_score_textrank.append(sentence_score/max_score)
            category["F11"].append(list_score_textrank)
        category["F11"] = flatten(category["F11"])
    return data


def f12_extraction(data):
    # sentence centrality
    # ratio unigram overlap sentence with overall unigram in doc
    for category in data:
        category['F12'] = []
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
            category['F12'].append(list_score_overlap)
        category["F12"] = flatten(category["F12"])
    return data

def compute_feature(data):
    data = f1_extraction(data)
    data = f2_extraction(data)
    data = f3_extraction(data)
    data = f5_extraction(data)
    data = f6_extraction(data)
    data = f7_extraction(data)
    data = f9_extraction(data)
    data = f10_extraction(data)
    data = f11_extraction(data)
    return data

def save_feature(data, precomputed=False, file_dir="analysis/feature_set.jsonl"):
    data = data if precomputed else compute_feature(data)
    selected_field = ["id", "F1", "F2", "F3", "F5", "F6", "F7", "F9", "F10", "F11", "F12",'gold_labels']
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

# Must run save_feature first
# Array look alike data
def save_array_data_for_model(file_dir="analysis/feature_set.jsonl", file_save="analysis/nested_data.txt"):
    data = []
    flatten = lambda l: [item for sublist in l for item in sublist]
    for line in open(file_dir, 'r'):
        data.append(json.loads(line))

    # Make gold_labels binary
    for doc in data:
        list_label = []
        for label in doc['gold_labels']:
            label = [1 if boolean==True else 0 for boolean in label]
            list_label.append(label)
        doc['gold_labels'] = flatten(list_label)

    #Flatten data
    for doc in data:
        for feat in doc:
            if isinstance(doc[feat], list):
                doc[feat] = flatten(doc[feat])
            else:
                continue

    # Save data into nested array
    data_transpose = []
    for doc in data:
        data_matrix = []
        for feat in doc:
            if isinstance(doc[feat], list):
                data_matrix.append(doc[feat])
            else:
                continue
        transpose_matrix = list(map(list, zip(*data_matrix)))
        data_transpose.append(transpose_matrix)
    nested_array = flatten(data_transpose)

    # Save file
    np.savetxt(file_save, nested_array, fmt='%s')

def demo():
    data = [open_dataset("dev", 1),open_dataset("train", 1), open_dataset("test", 1)]
    data = flatten(data)
    # data = pre_processed_all(data)
    print("F1")
    t1 = time.time()
    data = f1_extraction(data)
    print(data[0]["F1"])
    t2 = time.time()
    print('Elapsed time: {}'.format(timedelta(seconds=t2-t1)))
    print("F2")
    data = f2_extraction(data)
    print("F3")
    data = f3_extraction(data)
    print("F5")
    data = f5_extraction(data)
    print("F6")
    data = f6_extraction(data)
    print("F7")
    data = f7_extraction(data)
    print("F9")
    data = f9_extraction(data)
    #print(data[0])
    print("F10")
    data = f10_extraction(data)
    print("F11")
    data = f11_extraction(data)
    #print(data[0])
    print("F12")
    data = f12_extraction(data)
    #print(data[0])
    save_feature(data, precomputed=True)

if __name__ == "__main__":
    demo()

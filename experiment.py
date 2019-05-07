from data import open_dataset
from preprocessor import pre_processed_all
from feature_extraction import compute_feature

from model.lead3 import Lead3
from model.svm import SVM

from evaluation import Evaluator
from rouge_evaluation import Rouge
from log import Log

import numpy as np
import json
import time
from datetime import timedelta
from sklearn.svm import SVC

methods_ready = ["svm"]
LOG_FILE_NAME = "log_{}.txt"
PRECOMP_PREPROCESSED_FILE = "analysis/precomputed_dataset.jsonl"
FEATURE_SET_FILE = "analysis/feature_set.jsonl"

log = Log()
flatten = lambda l: [item for sublist in l for item in sublist]
feature_attr_name = ["F1", "F2", "F3", "F5", "F6", "F7", "F9", "F10", "F11", "F12"]
preprocessing_attr = ["stopped_paragraphs", "stemmed_paragraphs", "lemma_paragraphs", "word_tag"]

def lead3_experiment(test_data):
    log.write("Predicting with Lead3")
    for doc in test_data:
        doc["predicted_labels"] = {}
        doc["predicted_labels"]["lead3"] = Lead3.predict(doc)
    predicted_labels = flatten([flatten(i["predicted_labels"]["lead3"]) for i in test_data])
    return predicted_labels

def negative_sampling(matrix, label):
    label_set = set(label)
    true_data = np.where(label==True,)[0]
    false_data = np.where(label==False,)[0]
    n_true = len(true_data)
    n_false = len(false_data)

    selected_false_idx = np.random.choice(false_data, 2*n_true, replace=False)
    selected_data = np.concatenate((true_data, selected_false_idx))
    # new_matrix =
    new_matrix = matrix[selected_data]
    new_label = label[selected_data]
    return new_matrix, new_label

def svm_experiment(train_data, validation_data, test_data):
    conf = {"kernel": "linear_kernel", "degree":2, "sigma":50, "C":100}
    # merge train and validation
    log.write(conf)
    log.write("Preparing data training")
    # build feature matrix
    train_data = flatten([train_data, validation_data])
    train_feature_matrix = []
    train_label_vector = []
    for doc in train_data:
        # print(len(flatten(doc["paragraphs"])))
        # print(doc)
        for idx, sentences in enumerate(flatten(doc["paragraphs"])):
            sentence_feature = []
            for attr in feature_attr_name:
                sentence_feature.append(doc[attr][idx])
            train_feature_matrix.append(sentence_feature)
            train_label_vector.append(flatten(doc["gold_labels"])[idx])
    n_data = len(train_feature_matrix)
    split_length = int(n_data/20)
    offset = 0
    X = train_feature_matrix
    y = train_label_vector
    log.write("Preparing data testing")
    test_feature_matrix = []
    for doc in test_data:
        for idx, sentences in enumerate(flatten(doc["paragraphs"])):
            sentence_feature = []
            for attr in feature_attr_name:
                sentence_feature.append(doc[attr][idx])
            test_feature_matrix.append(sentence_feature)
    # predict test_data
    test_feature_matrix = np.array(test_feature_matrix)
    all_prediction = np.zeros((len(test_feature_matrix), 2))
    for i in range(1):
        train_feature_matrix = np.array(X[offset:(offset+split_length)])
        train_label_vector = np.array(y[offset:(offset+split_length)])
        offset += split_length
        # run_training
        train_feature_matrix, train_label_vector = negative_sampling(train_feature_matrix, train_label_vector)
        log.write("Training SVM")
        svm_clf = SVM(kernel=conf["kernel"], C=conf["C"], sigma=conf["sigma"])
        svm_clf.fit(train_feature_matrix, train_label_vector)
        t1 = time.time()
        log.write("Testing SVM")
        predicted_labels, val = svm_clf.predict(test_feature_matrix)
        t2 = time.time()
        print('Elapsed time: {}'.format(timedelta(seconds=t2-t1)))
        predicted_labels = [1 if i>-1 else 0 for i in predicted_labels]
        for idx, label in enumerate(predicted_labels):
            all_prediction[idx][label] += 1
    predicted_labels = np.argmax(all_prediction, axis=1)
    return predicted_labels

def run_experiment(train_data, validation_data, test_data, method):
    if method=="lead3":
        predicted_labels = lead3_experiment(test_data)
    elif method=="svm":
        predicted_labels = svm_experiment(train_data, validation_data, test_data)
    else:
        raise(InvalidMethod())
    return predicted_labels

def make_summary(doc, predicted_labels):
    selected_summary = []
    sentence_candidate = flatten(doc["paragraphs"])
    for idx, label in enumerate(predicted_labels):
        if label == 1:
            selected_summary.append(sentence_candidate[idx])
    return selected_summary

def get_data(fold, feature_data=None,  preprocessed_data=None):
    train_data = open_dataset("train", fold)
    val_data = open_dataset("dev", fold)
    test_data = open_dataset("test", fold)
    # print(len(feature_data))
    # print(len(flatten([train_data, val_data, test_data])))
    counter = 0
    for data_split in [train_data, val_data, test_data]:
        for doc in data_split:
            if doc["id"] == "1500977700-roberto-baggio-antara-dosa-dan-perjalanan-taubat-s":
                print(doc["paragraphs"][-1])
                print(len(flatten(doc["gold_labels"])))
            for attr in feature_attr_name:
                doc[attr] = feature_data[doc["id"]][attr]
            if preprocessed_data:
                for attr in preprocessing_attr:
                    doc[attr] = preprocessed_data[doc["id"]][attr]
    return train_data, val_data, test_data

def preprocessing_data(data):
    try:
        log.write("Open precomputed pre-processed dataset:")
        log.write(PRECOMP_PREPROCESSED_FILE)
        with open(PRECOMP_PREPROCESSED_FILE, "r") as f:
            data = []
            line = f.readline()
            while line:
                data.append(json.loads(line))
                line = f.readline()
    except:
        log.write("File not found, start preprocessing...")
        data = pre_processed_all(data)
    data = {i["id"]: i for i in data}
    return data

def feature_extraction(data):
    try:
        log.write("Open precomputed feature dataset:")
        log.write(FEATURE_SET_FILE)
        with open(FEATURE_SET_FILE, "r") as f:
            data = []
            line = f.readline()
            while line:
                data.append(json.loads(line))
                line = f.readline()
    except:
        log.write("File not found, start extraction...")
        data = compute_feature(list(data.values()))
    data = {i["id"]: i for i in data}
    return data

def main():
    # get precomputed file / compute preprocessing & feature
    data = [open_dataset("dev", 1),open_dataset("train", 1), open_dataset("test", 1)]
    # log.write("Preprocessing dataset...")
    log.write("Feature extraction...")
    # pre_processed_data = preprocessing_data(flatten(data))
    feature_data = feature_extraction(flatten(data))
    for fold in range(1,6):
        log.write("Get fold {} of IndoSum dataset".format(fold))
        train_data, val_data, test_data = get_data(fold, feature_data)
        for method in methods_ready:
            log.write("==================================")
            log.write("Prediction using {}".format(method))
            log.write("==================================")
            predicted_labels = run_experiment(train_data, val_data, test_data, method)

            log.write("Evaluating {}".format(method))

            gold_labels = flatten([flatten(i["gold_labels"]) for i in test_data])
            gold_labels = [1 if i else 0 for i in gold_labels]
            metric_evaluation = Evaluator()
            metric_evaluation.compute_all(gold_labels, predicted_labels)
            log.write("Confusion Matrix :")
            log.write(metric_evaluation.confusion_matrix)
            log.write("Accuracy     = %f"%metric_evaluation.accuracy)
            log.write("Precision    = %f"%metric_evaluation.precision)
            log.write("Recall       = %f"%metric_evaluation.recall)
            log.write("F1 Score     = %f"%metric_evaluation.f1_score)
            log.write("ROUGE evaluation")

            rouge_score = {"1": [], "2":[], "L":[]}
            rouge_precision = {"1": [], "2":[], "L":[]}
            rouge_recall = {"1": [], "2":[], "L":[]}

            sentence_offset = 0
            for doc in test_data:
                n_sentence = len(flatten(doc["paragraphs"]))
                selected_summary = make_summary(doc, predicted_labels[sentence_offset:sentence_offset+n_sentence])
                sentence_offset += n_sentence
                rouge_eval = Rouge().compute_all(doc["summary"], selected_summary)

                rouge_score["1"].append(rouge_eval.rouge_1_score)
                rouge_score["2"].append(rouge_eval.rouge_2_score)
                rouge_score["L"].append(rouge_eval.rouge_l_score)
                rouge_precision["1"].append(rouge_eval.rouge_1_precision)
                rouge_precision["2"].append(rouge_eval.rouge_2_precision)
                rouge_precision["L"].append(rouge_eval.rouge_l_precision)
                rouge_recall["1"].append(rouge_eval.rouge_1_recall)
                rouge_recall["2"].append(rouge_eval.rouge_2_recall)
                rouge_recall["L"].append(rouge_eval.rouge_l_recall)
            log.write("Average rouge performance :")
            log.write("ROUGE-1 score        = %f"%(sum(rouge_score["1"])/len(test_data)))
            log.write("ROUGE-1 precision    = %f"%(sum(rouge_precision["1"])/len(test_data)))
            log.write("ROUGE-1 recall       = %f"%(sum(rouge_recall["1"])/len(test_data)))
            log.write("ROUGE-2 score        = %f"%(sum(rouge_score["2"])/len(test_data)))
            log.write("ROUGE-2 precision    = %f"%(sum(rouge_precision["2"])/len(test_data)))
            log.write("ROUGE-2 recall       = %f"%(sum(rouge_recall["2"])/len(test_data)))
            log.write("ROUGE-L score        = %f"%(sum(rouge_score["L"])/len(test_data)))
            log.write("ROUGE-L precision    = %f"%(sum(rouge_precision["L"])/len(test_data)))
            log.write("ROUGE-L recall       = %f"%(sum(rouge_recall["L"])/len(test_data)))
    log.close()

if __name__ == "__main__":
    main()

from data import open_dataset
from preprocessor import pre_processed_all
from feature_extraction import compute_feature

from model.lead3 import Lead3
from model.svm import SVM

from evaluation import Evaluator
from rouge_evaluation import Rouge
from log import Log

import numpy as np

methods_ready = ["lead3"]
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

def svm_experiment(train_data, validation_data, test_data):
    conf = {"kernel": "linear_kernel", "degree":3, "sigma":5}
    svm_clf = SVM(kernel=conf["kernel"])
    # merge train and validation
    train_data = flatten([train_data, validation_data])
    # build feature matrix
    train_feature_matrix = []
    train_label_vector = []
    for doc in train_data:
        for idx, sentences in enumerate(flatten(doc["paragraphs"])):
            sentence_feature = []
            for attr in feature_attr_name:
                sentence_feature.append(doc[attr][idx])
            train_feature_matrix.append(sentence_feature)
            train_label_vector.append(flatten(doc["gold_labels"])[idx])
    train_feature_matrix = np.array(train_feature_matrix)
    train_label_vector = np.array(train_label_vector)
    # run_training
    test_feature_matrix = []
    svm_clf.fit(train_feature_matrix, train_label_vector)
    for doc in test_data:
        for idx, sentences in enumerate(flatten(doc["paragraphs"])):
            sentence_feature = []
            for attr in feature_attr_name:
                sentence_feature.append(doc[attr][idx])
            test_feature_matrix.append(sentence_feature)
    # predict test_data
    test_feature_matrix = np.array(test_feature_matrix)
    predicted_labels = svm_clf.predict(test_feature_matrix)
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

def get_data(fold, preprocessed_data=None, feature_data=None):
    train_data = open_dataset("train", fold)
    val_data = open_dataset("dev", fold)
    test_data = open_dataset("test", fold)
    for data_split in [train_data, val_data, test_data]:
        for doc in data_split:
            for attr in feature_attr_name:
                doc[attr] = feature_data["id"][attr]
            for attr in pre_processed_all:
                doc[attr] = preprocessed_data["id"][attr]
    return train_data, val_data, test_data

def preprocessing_data(data):
    try:
        log.write("Open precomputed pre-processed dataset:")
        log.write(PRECOMP_PREPROCESSED_FILE)
        with open(PRECOMP_PREPROCESSED_FILE, "r") as f:
            data = []
            line = datafile.readline()
            while line:
                data.append(json.loads(line))
                line = datafile.readline()
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
            line = datafile.readline()
            while line:
                data.append(json.loads(line))
                line = datafile.readline()
    except:
        log.write("File not found, start extraction...")
        data = compute_feature(data)
    data = {i["id"]: i for i in data}
    return data

def main():
    # get precomputed file / compute preprocessing & feature
    data = [open_dataset("dev", 1),open_dataset("train", 1), open_dataset("test", 1)]
    log.write("Preprocessing dataset...")
    pre_preprocessed_data = preprocessing_data(flatten(data))
    log.write("Feature extraction...")
    feature_data = feature_extraction(flatten(data))

    for fold in range(1,6):
        log.write("Get fold {} of IndoSum dataset".format(fold))
        train_data, val_data, test_data = get_data(fold, pre_preprocessed_data, feature_data)
        for method in methods_ready:
            predicted_labels = run_experiment(train_data, validation_data, test_data, method)

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

            for doc in test_data:
                selected_summary = make_summary(doc, predicted_labels)
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

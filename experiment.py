from data import open_dataset
from preprocessor import pos_tagger, stopword_remover, word_stemmer, word_lemmatizer
from feature_extraction import compute_feature
from model.lead3 import Lead3
from evaluation import Evaluator

method_ready = ["lead3"]
def main():
    flatten = lambda l: [item for sublist in l for item in sublist]
    data = [open_dataset("dev", 1),open_dataset("train", 1), open_dataset("test", 1)]
    all_data = flatten(data)
    data = data[2]
    print("Predicting with Lead3")
    for doc in data:
        doc["predicted_labels"] = {}
        doc["predicted_labels"]["lead3"] = Lead3.predict(doc)
    print("Evaluating Lead3")
    predicted_labels = flatten([flatten(i["predicted_labels"]["lead3"]) for i in data])
    gold_labels = flatten([flatten(i["gold_labels"]) for i in data])
    gold_labels = [1 if i else 0 for i in gold_labels]
    metric_evaluation = Evaluator()
    metric_evaluation.compute_all(gold_labels, predicted_labels)
    print("Confusion Matrix :")
    print(metric_evaluation.confusion_matrix)
    print("Accuracy     = %f"%metric_evaluation.accuracy)
    print("Precision    = %f"%metric_evaluation.precision)
    print("Recall       = %f"%metric_evaluation.recall)
    print("F1 Score     = %f"%metric_evaluation.f1_score)
    # print(predicted_labels)

if __name__ == "__main__":
    main()

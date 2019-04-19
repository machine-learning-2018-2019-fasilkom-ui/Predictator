from data import open_dataset
from preprocessor import pos_tagger, stopword_remover, word_stemmer, word_lemmatizer
from feature_extraction import compute_feature
from model.lead3 import Lead3
from evaluation import Evaluator
from rouge_evaluation import Rouge

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
    print("ROUGE evaluation")
    rouge_value = {"1": [], "2":[], "L":[]}
    for doc in data:
        selected_summary = []
        sentence_candidate = flatten(doc["paragraphs"])
        for idx, label in enumerate(flatten(doc["predicted_labels"]["lead3"])):
            if label == 1:
                selected_summary.append(sentence_candidate[idx])
        rouge_eval = Rouge().compute_all(doc["summary"], selected_summary)
        rouge_value["1"].append(rouge_eval.rouge_1_score)
        rouge_value["2"].append(rouge_eval.rouge_2_score)
        rouge_value["L"].append(rouge_eval.rouge_l_score)
    print("average rouge performance :")
    print("ROUGE-1  = %f"%(sum(rouge_value["1"])/len(data)))
    print("ROUGE-2  = %f"%(sum(rouge_value["2"])/len(data)))
    print("ROUGE-L  = %f"%(sum(rouge_value["L"])/len(data)))
    # print(predicted_labels)

if __name__ == "__main__":
    main()

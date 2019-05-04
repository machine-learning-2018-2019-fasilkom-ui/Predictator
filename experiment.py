from data import open_dataset
from preprocessor import pos_tagger, stopword_remover, word_stemmer, word_lemmatizer
from feature_extraction import compute_feature
from model.lead3 import Lead3
from evaluation import Evaluator
from rouge_evaluation import Rouge
from log import Log

method_ready = ["lead3"]
LOG_FILE_NAME = "log_{}.txt"

def main():
    log = Log()
    flatten = lambda l: [item for sublist in l for item in sublist]
    data = [open_dataset("dev", 1),open_dataset("train", 1), open_dataset("test", 1)]
    all_data = flatten(data)
    for fold in range(1,6):
        val_data = open_dataset("dev", fold)
        train_data = open_dataset("train", fold)
        test_data = open_dataset("test", fold)
        log.write("Predicting with Lead3")
        for doc in test_data:
            doc["predicted_labels"] = {}
            doc["predicted_labels"]["lead3"] = Lead3.predict(doc)
        log.write("Evaluating Lead3")
        predicted_labels = flatten([flatten(i["predicted_labels"]["lead3"]) for i in test_data])
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
            selected_summary = []
            sentence_candidate = flatten(doc["paragraphs"])
            for idx, label in enumerate(flatten(doc["predicted_labels"]["lead3"])):
                if label == 1:
                    selected_summary.append(sentence_candidate[idx])
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
        log.write("average rouge performance :")
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

class Evaluator:
    def __init__(self):
        self.confusion_matrix = {(i,j): 0 for i in range(2) for j in range(2)}
        self.accuracy = 0.0
        self.precision = 0.0
        self.recall = 0.0
        self.f1_score = 0.0

    def _make_confusion_matrix(self, truth, predicted):
        for truth_label in truth:
            for predicted_label in predicted:
                self.confusion_matrix[(truth_label), (predicted_label)] += 1

    def _accuracy(self):
        tp = self.confusion_matrix[(1,1)]
        tn = self.confusion_matrix[(0,0)]
        n_data = len(self.confusion_matrix.value)
        self.accuracy = 1.0*(tp+tn)/n_data

    def _precision(self):
        tp = self.confusion_matrix[(1,1)]
        fp = self.confusion_matrix[(0,1)]
        n_data = len(self.confusion_matrix.value)
        self.precision = 1.0*(tp+fp)/n_data

    def _recall(self):
        tp = self.confusion_matrix[(1,1)]
        fn = self.confusion_matrix[(1,0)]
        n_data = len(self.confusion_matrix.value)
        self.recall = 1.0*(tp+fn)/n_data


    def _f1_score(self):
        self.f1_score = 2/((1/self.recall)+(1/self.precision))

    def compute_all(self, truth, predicted):
        self._make_confusion_matrix(truth, predicted)
        self._accuracy()
        self._precision()
        self._recall()
        self._f1_score()

    def compute_rouge(self, choosen_sentence, reference_summary):
        pass

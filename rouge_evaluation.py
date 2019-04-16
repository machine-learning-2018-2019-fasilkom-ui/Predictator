class Rouge:
    self.rouge_1_score = 0.0
    self.rouge_2_score = 0.0
    self.rouge_l_precision = 0.0
    self.rouge_l_recall = 0.0
    self.rouge_l_f_score = 0.0

    def _lcs_length(seq1, seq2):
        pass

    def rouge_1(reference_summary, choosen_sentence):
        pass

    def rouge_2(reference_summary, choosen_sentence):
        pass

    def rouge_l(reference_summary, choosen_sentence):
        pass

    def compute_all(reference_summary, choosen_sentence):
        rouge_1(reference_summary, choosen_sentence)
        rouge_2(reference_summary, choosen_sentence)
        rouge_l(reference_summary, choosen_sentence)

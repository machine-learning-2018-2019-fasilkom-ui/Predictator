import numpy as np

class Rouge:
    _flatten = lambda l: [item for sublist in l for item in sublist]

    def __init__(self):
        self.rouge_1_score = 0.0
        self.rouge_2_score = 0.0
        self.rouge_l_score = 0.0
        self.beta = 1.0

    def _lcs_length(self, seq1, seq2):
        n = len(seq1)
        m = len(seq2)
        lcs_table = np.zeros((n+1,m+1))
        for i in range(1,n+1):
            for j in range(1, m+1):
                if seq1[i-1] == seq2[j-1]:
                    lcs_table[i][j] = lcs_table[i-1][j-1] + 1.0
                else:
                    lcs_table[i][j] = max(lcs_table[i][j-1], lcs_table[i-1][j])
        return lcs_table[n][m]

    def _build_bigram_vocabulary(self, sequences):
        bigram = []
        for sequence in sequences:
            bigram.append([(sequence[word], sequence[word+1]) for word in range(len(sequence)-1)])
        return set(Rouge._flatten(bigram))

    def rouge_1(self, reference_summary, choosen_sentence):
        reference_summary = Rouge._flatten(reference_summary)
        choosen_sentence = Rouge._flatten(choosen_sentence)
        reference_unigram = set(reference_summary)
        choosen_unigram = set(choosen_sentence)
        match_unigram = 0
        self.rouge_1_recall = 0.0
        self.rouge_1_precision = 0.0
        for ref_word in reference_unigram:
            for choosen_word in choosen_unigram:
                if ref_word == choosen_word:
                    match_unigram += 1
        self.rouge_1_recall = 1.0*match_unigram/len(reference_unigram)
        self.rouge_1_precision = 1.0*match_unigram/len(choosen_unigram)
        self.rouge_1_score = 2.0/((1.0/self.rouge_1_recall)+(1.0/self.rouge_1_precision))

    def rouge_2(self, reference_summary, choosen_sentence):
        reference_bigram = self._build_bigram_vocabulary(reference_summary)
        choosen_bigram = self._build_bigram_vocabulary(choosen_sentence)
        match_bigram = 0
        self.rouge_2_recall = 0.0
        self.rouge_2_precision = 0.0
        for ref_word in reference_bigram:
            for choosen_word in choosen_bigram:
                if ref_word == choosen_word:
                    match_bigram += 1
        self.rouge_2_recall = 1.0*match_bigram/len(reference_bigram)
        self.rouge_2_precision = 1.0*match_bigram/len(choosen_bigram)
        if match_bigram == 0:
            self.rouge_2_score = 0
        else:
            self.rouge_2_score = 2.0/((1.0/self.rouge_2_recall)+(1.0/self.rouge_2_precision))

    def rouge_l(self, reference_summary, choosen_sentence):
        reference_summary = Rouge._flatten(reference_summary)
        choosen_sentence = Rouge._flatten(choosen_sentence)
        lcs_len = self._lcs_length(reference_summary, choosen_sentence)
        self.rouge_l_recall = 1.0*lcs_len/len(reference_summary)
        self.rouge_l_precision = 1.0*lcs_len/len(choosen_sentence)
        self.rouge_l_score = ((1+self.beta**2)*self.rouge_l_recall*self.rouge_l_precision)/(self.rouge_l_recall+ ((self.beta**2)*self.rouge_l_precision))

    def compute_all(self, reference_summary, choosen_sentence):
        self.rouge_1(reference_summary, choosen_sentence)
        self.rouge_2(reference_summary, choosen_sentence)
        self.rouge_l(reference_summary, choosen_sentence)
        return self

class Rouge:
    self.rouge_1_score = 0.0
    self.rouge_2_score = 0.0
    self.rouge_l_precision = 0.0
    self.rouge_l_recall = 0.0
    self.rouge_l_f_score = 0.0
    flatten = lambda l: [item for sublist in l for item in sublist]

    def _lcs_length(self, seq1, seq2):
        pass

    def _build_bigram_vocabulary(sequences):
        bigram = []
        for sequence in a:
            bigram.append([(sequence[word], sequence[word+1]) for word in range(len(sequence)-1)])
        return set(Rouge.flatten(bigram))

    def rouge_1(self, reference_summary, choosen_sentence):
        reference_summary = Rouge.flatten(reference_summary)
        choosen_sentence = Rouge.flatten(choosen_sentence)
        reference_unigram = set(reference_summary)
        choosen_unigram = set(choosen_sentence)
        match_unigram = 0
        self.rouge_1_recall 0.0
        self.rouge_1_precision = 0.0
        for ref_word in reference_unigram:
            for choosen_word in choosen_unigram:
                if ref_word == choosen_word:
                    match_unigram += 1
        self.rouge_1_recall = 1.0*match_unigram/len(reference_unigram)
        self.rouge_1_precision = 1.0*match_unigram/len(choosen_unigram)
        self.rouge_1_score = 2.0/((1.0/self.rouge_1_recall)+(1.0/self.rouge_1_precision))

    def rouge_2(reference_summary, choosen_sentence):
        reference_bigram = self._build_bigram_vocabulary(reference_summary)
        choosen_bigram = self._build_bigram_vocabulary(choosen_sentence)
        match_bigram = 0
        self.rouge_2_recall 0.0
        self.rouge_2_precision = 0.0
        for ref_word in reference_bigram:
            for choosen_word in choosen_bigram:
                if ref_word == choosen_word:
                    match_bigram += 1
        self.rouge_2_recall = 1.0*match_bigram/len(reference_bigram)
        self.rouge_2_precision = 1.0*match_bigram/len(choosen_bigram)
        self.rouge_2_score = 2.0/((1.0/self.rouge_2_recall)+(1.0/self.rouge_2_precision))

    def rouge_l(reference_summary, choosen_sentence):
        pass

    def compute_all(reference_summary, choosen_sentence):

        rouge_1(reference_summary, choosen_sentence)
        rouge_2(reference_summary, choosen_sentence)
        rouge_l(reference_summary, choosen_sentence)

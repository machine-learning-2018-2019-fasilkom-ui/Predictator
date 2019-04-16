# Proyek Akhir Machine Learning

Kelompok  : Predictator
Member    :
  1. Ajmal Kurnia-1806169433
  2. Muhammad Fauzi-1806280533
  3. Pray Somaldo-1806280571

## Extractive Summarization Bahasa Indonesia

Dataset taken from [IndoSum](https://github.com/kata-ai/indosum)

### Feature
  - F1 : sentence similarity based on unigram overlap
  - F2 : paragraph similarity based on unigram overlap (unfinished)
  - F3 : unique formatting (unfinished)
  - F4 : cue important phrases (not used)
  - F5 : sum of TF-IDF (Term Frequency - Inverse Document Frequency)
  - F6 : title unigram overlap
  - F7 : sentence position in the paragraph
  - F8 : cue trivial phrases (not used)
  - F9 : proper noun word in sentence (unfinished)
  - F10 : sum of TF-ISF (Term Frequency - Inverse Sentence Frequency)
  - F11 : TextRank score (unfinished)

### Machine Learning Used
  - Baseline (lead3) => picking the first 3 sentences of the document as summary

### Reference

class Lead3:
    def __init__():
        pass

    def train():
        # NO training Process
        pass

    def predict(document):
        label = []
        sentence_counter = 0
        for paragraph in document["paragraphs"]:
            sentence_label = []
            for sentence in paragraph:
                sentence_label.append(1 if sentence_counter<3 else 0)
                sentence_counter += 1
            label.append(sentence_label)
        return label

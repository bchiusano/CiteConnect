import pickle
from sentence_transformers import CrossEncoder
import numpy as np
import spacy

def calculate_metrics(top_10, targets):
    print("Targets: ", targets)
    print("Top 10: ", top_10)

    hits = [t for t in targets if t in top_10]

    row_recall = len(hits) / len(targets) if len(targets) > 0 else 0
    row_precision = len(hits) / len(top_10) if len(top_10) > 0 else 0

    rank = 0
    for i, ecli in enumerate(top_10, 1):
        if ecli in targets:
            rank = 1 / i
            break

    return row_recall, row_precision, rank


def check_relevance():
    # get the answer from the top 10
    # check relevance (grade documents)
    # reprompt
    pass


def regenerate_if_none():
    # if the generation was shit then we regenerate passing to the top ten method the feedback to find other documents
    pass


if __name__ == '__main__':

    with open("structured_train_data_with_text.pkl", "rb") as f:
        train_data = pickle.load(f)

    # nlp = spacy.load("nl_core_news_md")

    # need to shuffle the data
    np.random.shuffle(train_data)

    model = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')




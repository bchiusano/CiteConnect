from test_rag_infloat_multilingual import LegalRAGSystem, clean_ecli
import pandas as pd
import os

DATA_DIR = "../data/"
letters_path = os.path.join(DATA_DIR, "Dataset Advice letters on objections towing of bicycles.xlsx")
df = pd.read_excel(letters_path)
data = df.dropna(subset=['ECLI', 'geanonimiseerd_doc_inhoud'])
letter_list = data['geanonimiseerd_doc_inhoud']
ecli_list = data['ECLI']


# after the get top 10
def load_one_letter():
    letter = letter_list[0]
    ecli = ecli_list[0]
    return letter, ecli


def calculate_metrics():

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


def give_scores():
    return [not_hit for not_hit in top_10 if not_hit not in targets]


if __name__ == "__main__":
    agentic_rag= LegalRAGSystem()
    text, citation = load_one_letter()
    found_raw = agentic_rag.get_top_10_for_letter(text)

    targets = [clean_ecli(e) for e in str(citation).replace(';', ',').split(',') if clean_ecli(e)]
    top_10 = [clean_ecli(f) for f, s in found_raw]

    recall, precision, rank_score = calculate_metrics()
    # TODO: we should also check if the hits are in the first positions of the top_10
    print("NOT HIT: ", give_scores())

    print("Recall: ", recall,
          "Precision: ", precision,
          "Rank Score:", rank_score)

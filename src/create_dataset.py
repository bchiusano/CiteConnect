from test_rag_infloat_multilingual import clean_ecli
import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
import numpy as np

DATA_DIR = "../data/"
letters_path = os.path.join(DATA_DIR, "Dataset Advice letters on objections towing of bicycles.xlsx")
df = pd.read_excel(letters_path)
data = df.dropna(subset=['ECLI', 'geanonimiseerd_doc_inhoud'])
letter_list = data['geanonimiseerd_doc_inhoud']
ecli_list = data['ECLI']


def get_ecli_data():
    e_df = pd.read_excel("../data/DATA ecli_nummers juni 2025 v1 (version 1).xlsx")
    return e_df.set_index('ecli_nummer')


def get_ecli_text(ecli_df, number):

    return ecli_df.loc[f"ECLI:{number}", 'ecli_tekst']


def prepare_training_data(train_size, random_seed):
    df = pd.read_excel(letters_path)
    data = df.dropna(subset=['ECLI', 'geanonimiseerd_doc_inhoud'])
    # ecli_df = get_ecli_data()
    print(len(data))
    query_target_data = []

    for idx, row in data.iterrows():

        targets = [clean_ecli(e) for e in str(row['ECLI']).replace(';', ',').split(',') if clean_ecli(e)]

        query = str(row['geanonimiseerd_doc_inhoud'])

        '''
        # if there is more than one target
        if targets:
            try:
                clean_letter = remove_ecli_from_letters(query, targets)
                for target in targets:

                    query_target_data.append({
                        'question': clean_letter,
                        'answer': get_ecli_text(ecli_df, target)
                    })
            except: pass
        '''
        if targets:
            clean_letter = remove_ecli_from_letters(query, targets)
            query_target_data.append({
                'query': clean_letter,
                'targets': targets
            })

    # split into train/test
    train, test = train_test_split(
        query_target_data,
        train_size=train_size,
        random_state=random_seed
    )

    print(f"Data split: {len(train)} train, {len(test)} test")

    return train, test


def create_example(query, ecli, relevance):
    return {'query': query, 'ecli': ecli, 'relevance': relevance}


def give_scores(top_10):
    return [not_hit for not_hit in top_10 if not_hit not in targets]


def remove_ecli_from_letters(text, citations):
    for ref in citations:
        text = str(text).replace(str(ref), "")
    return text


if __name__ == "__main__":

    structured_data = []
    ecli_data = get_ecli_data()
    DEBUG = False

    with open("candidate_cache.pkl", "rb") as f:
        candidates_cache = pickle.load(f)

    train, test = prepare_training_data(train_size=0.8, random_seed=42)

    #all_data = prepare_training_data(train_size=0.8, random_seed=42)
    #df = pd.DataFrame(all_data)
    #df.to_csv('question_answer.csv', index=False)

    indices = list(range(len(train)))
    np.random.shuffle(indices)

    for i, idx in enumerate(indices):

        query = train[idx]['query']
        targets = train[idx]['targets']

        # letters still have the ECLI mentioned
        clean_letter = remove_ecli_from_letters(query, targets)

        candidate_ecli = candidates_cache.get(idx, [])

        top_candidates = [c['ecli'] for c in candidate_ecli]
        incorrect = give_scores(top_candidates)

        if DEBUG:
            print("TARGET: ", targets)
            print("TOP_TEN", top_candidates)
            print("NOT HIT: ", incorrect)

        # create positive examples
        for target in targets:
            try:
                target_text = f"ECLI:{target}: " + get_ecli_text(ecli_data, target)
                structured_data.append(create_example(query=clean_letter, ecli=target_text, relevance=1))
            except:
                print(f"{target} not found")
                pass

        # create negative examples
        for wrong in incorrect:
            try:
                wrong_text = f"ECLI:{wrong}: " + get_ecli_text(ecli_data, wrong)
                structured_data.append(create_example(query=clean_letter, ecli=wrong_text, relevance=0))
            except:
                print(f"{wrong} not found")
                pass
    if DEBUG:
        for d in structured_data:
            print(f"ECLI: {d['ecli']}, Relevance: {d['relevance']}")

    with open('structured_train_data_with_text.pkl', 'wb') as f:
        pickle.dump(structured_data, f)
    print("saved data to disk")


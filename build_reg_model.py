import argparse
import csv
import os
import pickle

from nltk import TweetTokenizer
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from utils import read_data_into_DFs, create_df, get_results

"""
Builds model which takes ngram TF-IDF sparse matrixes as features, and saves model and results

Takes PATH to training data as input
"""


def main():
    # Create DFs
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    args = parser.parse_args()
    folder = args.input
    cwd = os.getcwd()
    DFs = read_data_into_DFs(folder)
    os.chdir(cwd)
    # Model and feature params
    vec_params = {
        "word_range": (1, 2),
        "word_min": 0.05,
        "word_max": 0.85,
        "char_range": (1, 6),
        "char_min": 0.001,
        "char_max": 1.0}
    model_params = {
        "C": 100000,
        "class_weight": "balanced",
        "intercept_scaling": 0.1,
        # "loss": "squared_hinge",
        "max_iter": 2000,
        "random_state": 42,
        "tol": 0.00001}
    # Prep data
    X_train, X_dev, Y_train, Y_dev = create_df(DFs[0], prepro=True)
    # X = create_df(DFs[1], train=False, prepro=True, demoji= False)
    # Y = DFs[1]['class']
    # model = LinearSVC(**model_params)
    # model = MultinomialNB()
    # model = RandomForestClassifier(random_state=42)
    model = LogisticRegression(**model_params)

    # Create TF-IDF matrixes
    word_Tfidf_vec = TfidfVectorizer(analyzer='word',
                                     tokenizer=TweetTokenizer().tokenize,
                                     ngram_range=vec_params['word_range'],
                                     min_df=vec_params['word_min'], max_df=vec_params["word_max"],
                                     use_idf=True,
                                     sublinear_tf=True)
    char_Tfidf_vec = TfidfVectorizer(analyzer='char',
                                     tokenizer=TweetTokenizer().tokenize,
                                     ngram_range=vec_params['char_range'],
                                     min_df=vec_params['char_min'], max_df=vec_params["char_max"],
                                     use_idf=True,
                                     sublinear_tf=True)
    column_transformer = ColumnTransformer(
        [('tfidf_word', word_Tfidf_vec, 'concat_1'),
         ('tfidf_char', char_Tfidf_vec, 'concat_2')],
        remainder='passthrough')
    X = column_transformer.fit_transform(X_train)
    XX_dev = column_transformer.transform(X_dev)
    # Fit model and dump
    model.fit(X, Y_train)
    filename = f"models/regression_model_es.sav"
    os.chdir(cwd)
    pickle.dump(model, open(filename, 'wb'))
    pickle.dump(column_transformer, open("models/column_transformer_es.sav", 'wb'))

    # Record results
    predictions = model.predict(XX_dev)
    accuracy, f_measures = get_results(Y_dev, predictions, "en")
    with open("en_runs.csv", "a", newline='') as file:
        writer = csv.writer(file, delimiter=",")

        writer.writerow(["SVC", accuracy, f_measures[0], f_measures[1]])
        writer.writerow(vec_params.values())
        writer.writerow(model_params.values())


if __name__ == "__main__":
    main()

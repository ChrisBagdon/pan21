import argparse
import os
import pickle

import numpy as np
from sklearn.svm import LinearSVC

from utils import read_data_into_DFs, data_for_bert, get_berts_opinion, create_df

"""
Builds, trains, and saves Meta-classifer from features generated by previously saved regression and 
RoBERTa models.

Takes PATH to training data as argument
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    args = parser.parse_args()
    folder = args.input
    # Create DFs
    cwd = os.getcwd()
    DFs = read_data_into_DFs(folder)
    os.chdir(cwd)
    # Set model hyper params
    model_params = {
        "C": 0.017,
        "class_weight": None,
        "intercept_scaling": 5,
        "loss": "hinge",
        "max_iter": 2000,
        "random_state": 42,
        "tol": 5}
    # Preprocess training data
    X_en = create_df(DFs[1], train=False, prepro=True)
    Y = DFs[1]['class']
    column_transformer_en = pickle.load(open("models/column_transformer_es.sav", 'rb'))
    model_input_en = column_transformer_en.transform(X_en)
    # Load regression model and get probs
    reg_model_en = pickle.load(open("models/regression_model_es.sav", "rb"))
    probs_en = reg_model_en.predict_proba(model_input_en)
    # Prep training data for BERT and get BERTs probs
    data_bert_en = data_for_bert(DFs[1], train=False)
    berts_opinion_en = get_berts_opinion(data_bert_en, "models/bert/es/best_model")
    # Combine outputs for meta classifier
    X_train_combo = np.concatenate([probs_en, berts_opinion_en], axis=1)
    # Build Meta Classifier, fit, and save model
    model = LinearSVC(**model_params)
    model.fit(X_train_combo, Y)
    pickle.dump(model, open("models/svm_model_combo_es.sav", 'wb'))


if __name__ == "__main__":
    main()

import argparse
import os
import pickle

import numpy as np

from utils import get_berts_opinion, data_for_bert, write_xml, create_df, \
    read_data_into_DF

"""
Uses saved models to make predictions on input XMLs and create XML output of predictions.

Takes PATH to input folder and PATH to output location as arguments.
"""
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", required=True)
parser.add_argument("-o", "--output", required=True)

args = parser.parse_args()


def main():
    # Gather Inputs
    cwd = os.getcwd()
    DF_en = read_data_into_DF(args.input, "en")
    DF_es = read_data_into_DF(args.input, "es")
    DFs = [DF_en, DF_es]
    os.chdir(cwd)

    # English

    # Prep data for BERT and get BERT's output
    text_for_bert_en = data_for_bert(DFs[0], train=False)
    berts_opinion_en = get_berts_opinion(text_for_bert_en, "models/bert/en/best_model")
    # Prep data for LR and get LR output
    X_en = create_df(DFs[0], train=False, prepro=True, demoji=False)
    reg_model_en = pickle.load(open("models/regression_model_en.sav", "rb"))
    column_transformer_en = pickle.load(open("models/column_transformer_en.sav", 'rb'))
    model_input_en = column_transformer_en.transform(X_en)
    probs_en = reg_model_en.predict_proba(model_input_en)
    # Run BERT and LR output through Meta Classifer
    X_combo_en = np.concatenate([berts_opinion_en, probs_en], axis=1)
    svm_model_en = pickle.load(open("models/svm_model_combo_en.sav", "rb"))
    predictions_en = svm_model_en.predict(X_combo_en)
    os.chdir(args.output)
    # Save to PAN21 standard XMLs
    if not os.path.exists("en"):
        os.mkdir("en")
        os.mkdir("es")
    for i, prediction in enumerate(predictions_en):
        id = DFs[0].iloc[i]['authorID']
        write_xml(id, 'en', prediction, f"en/{id}.xml")

    # Spanish
    os.chdir(cwd)
    text_for_bert_es = data_for_bert(DFs[1], train=False)
    berts_opinion_es = get_berts_opinion(text_for_bert_es, "models/bert/es/best_model")

    X_es = create_df(DFs[1], train=False, prepro=True, demoji=False)
    reg_model_es = pickle.load(open("models/regression_model_es.sav", "rb"))
    column_transformer_es = pickle.load(open("models/column_transformer_es.sav", 'rb'))
    model_input_es = column_transformer_es.transform(X_es)
    probs_es = reg_model_es.predict_proba(model_input_es)

    X_combo_es = np.concatenate([berts_opinion_es, probs_es], axis=1)
    svm_model_es = pickle.load(open("models/svm_model_combo_es.sav", "rb"))
    predictions_es = svm_model_es.predict(X_combo_es)
    os.chdir(args.output)
    for i, prediction in enumerate(predictions_es):
        id = DFs[1].iloc[i]['authorID']
        write_xml(id, 'es', prediction, f"es/{id}.xml")


if __name__ == "__main__":
    main()

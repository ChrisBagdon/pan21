import argparse
import pickle

from nltk import TweetTokenizer
from simpletransformers.classification import ClassificationArgs
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

import utils

"""
Deprecated main program file. Decided to use individual scripts for each step of the system.
"""
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input Directory Path", required=True)
# parser.add_argument("-o", "--output", help="Ouput Directory Path", required=True)
args = parser.parse_args()


def build_fit_predict(type, params, xy, lang, save=False):
    if type == 'LinearSVC':
        model = LinearSVC(**params)

    model.fit(xy[0], xy[2])
    if save:
        filename = f"models/regression_model_{lang}.sav"
        pickle.dump(model, open(filename, 'wb'))
    return model.predict(xy[1])


def build_fit_predict_bert(model_params, vec_params, xy, lang, save=False):
    model = LinearSVC(**model_params)
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

    pipe = Pipeline([('tfidf', column_transformer),
                     ('classify', model)])

    pipe.fit(xy[0], xy[2])
    X_dev = column_transformer.transform(xy[1])
    if save:
        filename = f"models/regression_model_{lang}.sav"
        pickle.dump(model, open(filename, 'wb'))
    return model_LSTM.predict(X_dev)


def main():
    DFs = utils.read_data_into_DFs(args.input)

    # EN dataset
    vec_params_en = {
        "word_range": (1, 2),
        "word_min": 0.05,
        "word_max": 0.85,
        "char_range": (1, 6),
        "char_min": 0.001,
        "char_max": 1.0}

    xy_en = utils.create_xy(DFs[0], vec_params_en, concat=True, preproc=False)

    params_en = {
        "C": 22458.78034,
        "class_weight": "balanced",
        "intercept_scaling": 0.877,
        "loss": "hinge",
        "max_iter": 2000,
        "random_state": 42,
        "tol": 0.00014376187,
    }
    prediction_en = build_fit_predict('LinearSVC', params_en, xy_en, "en")
    print("Base TF-IDF Model Results\n")
    utils.get_results(prediction_en, xy_en[3], "English")

    bert_model_args = ClassificationArgs()
    bert_model_args.num_train_epochs = 50
    bert_model_args.reprocess_input_data = True
    bert_model_args.overwrite_output_dir = True
    bert_model_args.evaluate_during_training = True
    bert_model_args.manual_seed = 4
    bert_model_args.use_multiprocessing = True
    bert_model_args.train_batch_size = 16
    bert_model_args.eval_batch_size = 8
    bert_model_args.labels_list = [0, 1]
    bert_model_args.learning_rate = 5.245514797925528e-05
    bert_model_args.wandb_project = "Sweep Results"
    bert_model_args.use_early_stopping = True
    bert_model_args.early_stopping_delta = 0.001
    bert_model_args.early_stopping_metric = "mcc"
    bert_model_args.early_stopping_metric_minimize = False
    bert_model_args.early_stopping_patience = 5
    bert_model_args.evaluate_during_training_steps = 30
    bert_model_args.evaluate_during_training_verbose = True
    # 5.202081638426634e-05
    bert_train_en, bert_dev_en = utils.data_for_bert(DFs[0])
    # utils.create_bert_model(DFs[0], bert_model_args)
    berts_xy = utils.create_xy_bert(DFs[0], [
        utils.get_berts_opinion(bert_train_en, "outputs/best_model"),
        utils.get_berts_opinion(bert_dev_en, "outputs/best_model")])
    preds_Bert_en = build_fit_predict_bert(params_en, vec_params_en, berts_xy, "en")

    print("TF-IDF with Bert results:\n")
    utils.get_results(preds_Bert_en, berts_xy[3], "English")

    # ES dataset
    vec_params_es = {
        "word_range": (1, 2),
        "word_min": 0.001,
        "word_max": 1.0,
        "char_range": (2, 6),
        "char_min": 0.001,
        "char_max": 0.75}

    xy_es = utils.create_xy(DFs[0], vec_params_es, concat=True, preproc=True)

    params_es = {
        "C": 8701.12835,
        "class_weight": None,
        "intercept_scaling": 9.095745,
        "loss": "squared_hinge",
        "max_iter": 2000,
        "random_state": 42,
        "tol": 0.0000017355634, }

    prediction_es = build_fit_predict('LinearSVC', params_es, xy_es, "es")

    utils.get_results(prediction_es, xy_es[3], "Spanish")


if __name__ == "__main__":
    main()

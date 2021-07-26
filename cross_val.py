import logging
import os

import numpy as np
import pandas as pd
from nltk import TweetTokenizer
from simpletransformers.classification import ClassificationModel
from simpletransformers.config.model_args import ClassificationArgs
from sklearn import metrics
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from utils import read_data_into_DFs, create_df, data_for_bert

"""
Performs 10 fold cross validation on entire System; SVM, BERT, and meta classifier.

Put together last minute before PAN deadline. Hyper parameters must be manually changed for English or Spanish

Prints individual and total average scores.
"""

# Prep data and create folds
scores_bert = []
scores_SVM = []
scores_meta = []
cwd = os.getcwd()
folder = "../pan21-author-profiling-training-2021-03-14"
DFs = read_data_into_DFs(folder)
# DFs[0] for English, DFs[1] for Spanish
DF1 = DFs[0]
DF1 = DF1.sample(frac=1).reset_index(drop=True)
fold = [DF1[:20], DF1[20:40], DF1[40:60], DF1[60:80], DF1[80:100],
        DF1[100:120], DF1[120:140], DF1[140:160], DF1[160:180], DF1[180:200]]
train = []
dev = []
for i, testi in enumerate(fold):
    train.append(fold[:i] + fold[i + 1:])
    dev.append(testi)

# Run each fold
for x in range(10):
    DF = train[x][0]
    for i, group in enumerate(train[x]):
        if i == 0:
            continue
        DF = pd.concat([DF, group], axis=0)
    DF = pd.concat([DF, dev[x]])

    vec_params = {
        "word_range": (1, 2),
        "word_min": 0.05,
        "word_max": 0.85,
        "char_range": (1, 6),
        "char_min": 0.001,
        "char_max": 1.0}
    model_params = {
        "C": 100000,
        "class_weight": None,
        "intercept_scaling": 0.1,
        # "loss": "squared_hinge",
        "max_iter": 2000,
        "random_state": 42,
        "tol": 0.0001537}

    # Split data
    X_train, X_dev, Y_train, Y_dev = create_df(DF, prepro=True)

    # Logistic Regression
    model_LR = LogisticRegression(**model_params)

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
                     ('classify', model_LR)])
    pipe.fit(X_train, Y_train)
    X_dev = column_transformer.transform(X_dev)
    X_train = column_transformer.transform(X_train)
    proba_LR_train = model_LR.predict_proba(X_train)
    proba_LR_dev = model_LR.predict_proba(X_dev)
    preds_LR = model_LR.predict(X_dev)
    scores_SVM.append(metrics.accuracy_score(Y_dev, preds_LR))
    # RoBerta
    logging.basicConfig(level=logging.INFO)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.WARNING)
    model_args = ClassificationArgs()
    model_args.learning_rate = 0.0000284
    model_args.num_train_epochs = 3
    model_args.output_dir = "models/bert"
    model_args.wandb_project = "Bert Cross Models 1"

    model_args.reprocess_input_data = True
    model_args.overwrite_output_dir = True
    model_args.evaluate_during_training = True
    model_args.manual_seed = 4
    model_args.use_multiprocessing = True
    model_args.train_batch_size = 8
    model_args.eval_batch_size = 4
    model_args.labels_list = [0, 1]
    model_args.evaluate_during_training_steps = 1000
    model_args.evaluate_during_training_verbose = True
    model_args.sliding_window = True
    model_args.train_custom_parameters_only = False
    model_args.gradient_accumulation_steps = 2
    model_args.reprocess_input_data = True
    model_args.no_save = True
    model_args.no_cache = True
    model_args.save_best_model = False
    model_args.special_tokens_list = ["#EMOJI#", "#HASHTAG#", "#USER#", "#URL#"]
    model_args.weight_decay = 0.1
    # Create a TransformerModel
    model_bert = ClassificationModel(
        "roberta",
        # "skimai/spanberta-base-cased",
        "roberta-base",
        use_cuda=True,
        args=model_args, )
    train_df1, eval_df1 = data_for_bert(DF, train=True)
    model_bert.train_model(train_df1, eval_df=eval_df1, accuracy=lambda truth, predictions: accuracy_score(
        truth, [round(p) for p in predictions]), )
    bert_dev = data_for_bert(DF, train=False)
    outputs_train = model_bert.predict(bert_dev[:160])
    outputs_dev = model_bert.predict(bert_dev[160:])
    berts_opinion_train = np.array([np.median(x, axis=0) for x in outputs_train[1]])
    berts_opinion_dev = np.array([np.median(x, axis=0) for x in outputs_dev[1]])
    # Eval and save output
    model_bert.eval_model(eval_df1)
    scores_bert.append(model_bert.results["accuracy"])
    # English Meta-classifer
    model_params = {
        "C": 0.015,
        "class_weight": None,
        "intercept_scaling": 5,
        "loss": "hinge",
        "max_iter": 2000,
        "random_state": 42,
        "tol": 0.5}
    X_en = create_df(DFs[1], train=False, prepro=True)
    Y = DF['class']

    X_train_combo = np.concatenate([proba_LR_train, berts_opinion_train], axis=1)
    X_dev_combo = np.concatenate([proba_LR_dev, berts_opinion_dev], axis=1)
    model_SVM = LinearSVC(**model_params)
    # Fit, Predict, and Save predictions
    model_SVM.fit(X_train_combo, Y_train)
    predictions = model_SVM.predict(X_dev_combo)

    scores_meta.append(metrics.accuracy_score(Y_dev, predictions))
    cross_meta_accuracy = np.average(scores_meta)
    print(f"Average Accuracy: {cross_meta_accuracy} Runs: {x + 1}")
for x in range(10):
    print(f"SVM: {scores_SVM[x]}\nBERT: {scores_bert[x]}\nMeta: {scores_meta[x]}")
print(f"Averages:\nSVM: {np.average(scores_SVM)}\nBERT: {np.average(scores_bert)}\n"
      f"META: {np.average(scores_meta)}")

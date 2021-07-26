import codecs
import glob
import logging
import os
import re
import xml.etree.ElementTree as ET

import emoji
import numpy as np
import pandas as pd
from nltk.tokenize import TweetTokenizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from simpletransformers.classification import ClassificationModel
from sklearn import metrics

"""
Misc. helper methods used throughout system
"""


def read_data_into_DFs(folder):
    """
    Reads data into a pandas DF for en and es. Features are lang, class, authorID, individual tweets
    and a fully concatted tweet.
    :param folder: Path to pan21-author-profiling-training-2021-03-14 dir
    :return: list of two pandas DF
    """
    cwd = os.getcwd()
    os.chdir(folder)
    en_xmls = glob.glob('en/*.xml')
    es_xmls = glob.glob('es/*.xml')
    os.chdir(cwd)
    en_dir = folder
    en_authors_DFs = []
    en_texts = pd.DataFrame

    for xml in en_xmls:
        xtree = ET.parse(xml)
        root = xtree.getroot()
        root = xtree.getroot()
        concat_tweet = ""
        authorDic = root.attrib
        authorDic['authorID'] = xml[3:-4]

        for x, doc in enumerate(root.iter('document')):
            tweet = "tweet" + str(x)
            authorDic[tweet] = doc.text
            concat_tweet = concat_tweet + doc.text + " "
        authorDic['concat'] = concat_tweet
        authorDF = pd.DataFrame(authorDic, index=[0])
        en_authors_DFs.append(authorDF)

    en_texts = pd.concat(en_authors_DFs, axis=0)

    es_authors_DFs = []
    es_texts = pd.DataFrame

    for xml in es_xmls:
        xtree = ET.parse(xml)
        root = xtree.getroot()
        concat_tweet = ""
        authorDic = root.attrib
        authorDic['authorID'] = xml[3:-4]

        for x, doc in enumerate(root.iter('document')):
            tweet = "tweet" + str(x)
            authorDic[tweet] = doc.text
            concat_tweet = concat_tweet + doc.text + " "
        authorDic['concat'] = concat_tweet
        authorDF = pd.DataFrame(authorDic, index=[0])
        es_authors_DFs.append(authorDF)

    es_texts = pd.concat(es_authors_DFs, axis=0)

    return en_texts


def read_data_into_DF(folder, lang):
    """
    Reads data into a pandas DF for en and es. Features are lang, class, authorID, individual tweets
    and a fully concatted tweet.
    :param folder: Path to pan21-author-profiling-training-2021-03-14 dir
    :return: list of two pandas DF
    """
    cwd = os.getcwd()
    os.chdir(folder)
    en_xmls = glob.glob(f'{lang}/*.xml')
    os.chdir(cwd)
    en_dir = folder
    en_authors_DFs = []
    en_texts = pd.DataFrame

    for xml in en_xmls:
        print(xml)
        tree = ET.parse(
            codecs.open(os.path.join(en_dir, xml), encoding="UTF-8"),
            parser=ET.XMLParser(encoding='UTF-8'),
        )
        root = tree.getroot()
        concat_tweet = ""
        authorDic = root.attrib
        authorDic['authorID'] = xml[3:-4]

        for x, doc in enumerate(root.iter('document')):
            tweet = "tweet" + str(x)
            authorDic[tweet] = doc.text
            concat_tweet = concat_tweet + doc.text + " "
        authorDic['concat'] = concat_tweet
        authorDF = pd.DataFrame(authorDic, index=[0])
        en_authors_DFs.append(authorDF)

    en_texts = pd.concat(en_authors_DFs, axis=0)

    return en_texts


def preprocess_tweets(DF, demoji=False):
    pro_DF = []
    for tweet in DF:
        pro_tweet = tweet
        if demoji:
            pro_tweet = emoji.demojize(tweet)
            pro_tweet = re.sub(r"(:[A-Za-z0-9_-]+:)", " #EMOJI# ", pro_tweet)
        tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True)
        words = tokenizer.tokenize(pro_tweet)
        detokenizer = TreebankWordDetokenizer()
        pro_tweet = detokenizer.detokenize(words)
        pro_DF.append(pro_tweet)
    return pro_DF


def preprocess_single_tweet(tweet):
    pro_tweet = emoji.demojize(tweet)
    pro_tweet = re.sub(r"(:[A-Za-z0-9_-]+:)", " #EMOJI# ", pro_tweet)
    tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True)
    words = tokenizer.tokenize(pro_tweet)
    detokenizer = TreebankWordDetokenizer()
    return detokenizer.detokenize(words)


def get_results(Y_dev, Y_pred, lang):
    accuracy = metrics.accuracy_score(Y_dev, Y_pred)
    precisions, recalls, f_measures, supports = metrics.precision_recall_fscore_support(Y_dev, Y_pred)

    print(f"Language: {lang}\nAccuracy: {accuracy}\nPrecisions: {precisions[0]}, {precisions[1]}\n"
          f"Recalls: {recalls[0]}, {recalls[1]}\nF Scores: {f_measures[0]}, {f_measures[1]}")
    return [accuracy, f_measures]


def write_xml(id, lang, label, file):
    tmpl = """
    <author id="{author_id}"
        lang="{lang}"
        type="{label}"
    />"""
    value = tmpl.format(author_id=id, lang=lang, label=label, )
    with open(file, "w") as f:
        f.write(value)


def create_df(DF, train=True, prepro=False, demoji=False):
    if train:
        concats = DF['concat'].tolist()
        if prepro:
            concats = preprocess_tweets(concats, demoji)
        Y = DF['class']

        X_train = pd.DataFrame({'concat_1': concats[:160],
                                'concat_2': concats[:160]})
        X_dev = pd.DataFrame({'concat_1': concats[160:],
                              'concat_2': concats[160:]})
        Y_train = Y[:160]
        Y_dev = Y[160:]

        return [X_train, X_dev, Y_train, Y_dev]
    else:
        concats = DF['concat'].tolist()
        if prepro:
            concats = preprocess_tweets(concats, demoji)
        return pd.DataFrame({'concat_1': concats,
                             'concat_2': concats})


def data_for_bert(DF, train=True):
    texts = []
    for index, row in DF.iterrows():
        if train:
            texts.append([preprocess_single_tweet(row["concat"]), int(row["class"])])
        else:
            texts.append([preprocess_single_tweet(row["concat"])])

    new_DF = pd.DataFrame(texts)
    if train:
        new_DF.columns = ["text", "labels"]
        train = new_DF.iloc[:160]
        dev = new_DF.iloc[160:]
        return [train, dev]
    else:
        return texts


def create_bert_model(train, dev, model_args):
    logging.basicConfig(level=logging.INFO)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.WARNING)

    model = ClassificationModel(
        "roberta",
        "roberta-base",
        use_cuda=True,
        args=model_args)

    model.train_model(train, eval_df=dev)


def get_berts_opinion(DF, model):
    model = ClassificationModel(
        "roberta",
        model)
    result, model_outputs = model.predict(DF)
    # combined = np.array([np.sum(x, axis=0) for x in model_outputs])
    # combined = np.array([np.mean(x, axis=0) for x in model_outputs])
    combined = np.array([np.median(x, axis=0) for x in model_outputs])

    return combined


def create_df_bert(DF, berts_opinion, train=True, prepro=False):
    if train:
        concats = DF['concat'].tolist()
        if prepro:
            concats = preprocess_tweets(concats, demoji=True)
        Y = DF['class']

        X_train = pd.DataFrame({'concat_1': concats[:160],
                                'concat_2': concats[:160],
                                'bert_x': berts_opinion[:160, 0],
                                "bert_y": berts_opinion[:160, 1]})
        X_dev = pd.DataFrame({'concat_1': concats[160:],
                              'concat_2': concats[160:],
                              'bert_x': berts_opinion[160:, 0],
                              "bert_y": berts_opinion[160:, 1]})
        Y_train = Y[:160]
        Y_dev = Y[160:]

        return [X_train, X_dev, Y_train, Y_dev]
    else:
        concats = DF['concat'].tolist()
        if prepro:
            concats = preprocess_tweets(concats, demoji=True)
        return pd.DataFrame({'concat_1': concats,
                             'concat_2': concats,
                             'bert_x': berts_opinion[:, 0],
                             "bert_y": berts_opinion[:, 1]})

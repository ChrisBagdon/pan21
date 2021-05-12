import os
import glob

import emoji
import pandas as pd
import xml.etree.ElementTree as ET
import re

from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn import metrics
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack


def read_data_into_DFs(folder):
    """
    Reads data into a pandas DF for en and es. Features are lang, class, authorID, individual tweets
    and a fully concatted tweet.
    :param folder: Path to pan21-author-profiling-training-2021-03-14 dir
    :return: list of two pandas DF
    """
    os.chdir(folder)
    en_xmls = glob.glob('en/*.xml')
    es_xmls = glob.glob('es/*.xml')

    en_authors_DFs = []
    en_texts = pd.DataFrame

    for xml in en_xmls:
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

    return [en_texts, es_texts]

def get_feats(texts, vec_params):

    #texts = utils.preprocess_tweets(texts)
    feats = []
    word_Tfidf_vec = TfidfVectorizer(analyzer='word',
                                   tokenizer=TweetTokenizer().tokenize,
                                   ngram_range=vec_params['word_range'],
                                   min_df=vec_params['word_min'], max_df=vec_params["word_max"],
                                   use_idf=True,
                                   sublinear_tf=True)
    feats.append(('wgram', word_Tfidf_vec))

    char_Tfidf_vec = TfidfVectorizer(analyzer='char',
                                   tokenizer=TweetTokenizer().tokenize,
                                   ngram_range=vec_params['char_range'],
                                   min_df=vec_params['char_min'], max_df=vec_params["char_max"],
                                   use_idf=True,
                                   sublinear_tf=True)
    feats.append(('cgram', char_Tfidf_vec))
    vectorizer = Pipeline([("feats", FeatureUnion(feats))])

    return vectorizer.fit_transform(texts)

def preprocess_tweets(DF):
    pro_DF = []
    for tweet in DF:
        pro_tweet = emoji.demojize(tweet)
        #pro_tweet = re.sub(r"(:[A-Za-z0-9_-]+:)", " ##emoji ", pro_tweet)
        tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True)
        words = tokenizer.tokenize(pro_tweet)
        detokenizer = TreebankWordDetokenizer()
        pro_tweet = detokenizer.detokenize(words)
        pro_DF.append(pro_tweet)
    return pd.Series(pro_DF)

def create_xy(DF, vec_params, concat=True):
    y = DF['class']
    if concat:
        return train_test_split(get_feats(DF['concat'], vec_params), y, test_size=0.20, random_state=42)
    else:
        return train_test_split(vec_per_tweet(DF, vec_params), y, test_size=0.20, random_state=42)

def vec_per_tweet(DF, vec_params):
    indv_tweets = DF.drop(['lang', 'class', 'authorID'], axis=1)
    data_for_model = []
    for column in indv_tweets:
        data_for_model.append(get_feats(indv_tweets[column], vec_params))

    return hstack(data_for_model)


def get_results(Y_dev, Y_pred, lang):

    accuracy = metrics.accuracy_score(Y_dev, Y_pred)
    precisions, recalls, f_measures, supports = metrics.precision_recall_fscore_support(Y_dev, Y_pred)

    print(f"Language: {lang}\nAccuracy: {accuracy}\nPrecisions: {precisions[0]}, {precisions[1]}\n"
          f"Recalls: {recalls[0]}, {recalls[1]}\nF Scores: {f_measures[0]}, {f_measures[1]}")
import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET


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
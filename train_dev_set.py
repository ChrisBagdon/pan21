import pandas as pd
import os
import glob
import argparse
import shutil
import random

"""
Creates dev and training sets and matching truth.txt

Args: -i path to data sets directory, -s size of data set
"""

if __name__ == "__main__":

    owd = os.getcwd()

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", help="Input Directory Path", required=True)
    parser.add_argument("-s", "--size", help="Size of data set", required=True)
    args = parser.parse_args()

    split_size = int(int(args.size) * 0.8)

    en_input = args.dir + '/en'
    os.chdir(en_input)

    truth_data_en = pd.read_csv('truth.txt', sep=':::', names=['author_id', 'spreader'])
    en_full_truth = open("truth.txt")

    os.mkdir("entrain")
    os.mkdir("endev")

    en_truth_train = open("entrain/truth.txt", "x")
    en_truth_dev = open("endev/truth.txt", "x")

    xmls = glob.glob("*.xml")
    random.shuffle(xmls)
    dev_xml = xmls[split_size:]
    train_xml = xmls[:split_size]

    for file in train_xml:
        target = "entrain/" + file
        shutil.copyfile(file, target)
        id = file[:-4]
        type = str(truth_data_en.loc[truth_data_en['author_id'] == id, 'spreader'].iloc[0])
        en_truth_train.write(id+':::'+type+ '\n')
    for file in dev_xml:
        target = "endev/" + file
        shutil.copyfile(file, target)
        id = file[:-4]
        type = str(truth_data_en.loc[truth_data_en['author_id'] == id, 'spreader'].iloc[0])
        en_truth_dev.write(id+':::'+type+ '\n')

    en_truth_dev.close()
    en_truth_train.close()
    en_full_truth.close()

    # ES Begin
    os.chdir(owd)

    es_input = args.dir + '/es'

    os.chdir(es_input)
    os.mkdir("estrain")
    os.mkdir("esdev")

    truth_data_es = pd.read_csv('truth.txt', sep=':::', names=['author_id', 'spreader'])
    es_full_truth = open("truth.txt")
    es_truth_train = open("estrain/truth.txt", "x")
    es_truth_dev = open("esdev/truth.txt", "x")

    xmls_es = glob.glob("*.xml")
    random.shuffle(xmls_es)
    dev_xml2 = xmls_es[split_size:]
    train_xml2 = xmls_es[:split_size]

    for file in train_xml2:
        target = "estrain/" + file
        shutil.copyfile(file, target)
        id = file[:-4]
        type = str(truth_data_es.loc[truth_data_es['author_id'] == id, 'spreader'].iloc[0])
        es_truth_train.write(id + ':::' + type+ '\n')
    for file in dev_xml2:
        target = "esdev/" + file
        shutil.copyfile(file, target)
        id = file[:-4]
        type = str(truth_data_es.loc[truth_data_es['author_id'] == id, 'spreader'].iloc[0])
        es_truth_dev.write(id + ':::' + type + '\n')
    es_truth_dev.close()
    es_truth_train.close()
    es_full_truth.close()







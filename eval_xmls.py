import argparse
import glob
import os
import xml.etree.ElementTree as ET

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

"""
Compares XML output with truth.txt to print classification reports and confusion matrices. 

Argument: -i "path/to/xml/files"
truth.txt must be in same directory as XMLs
"""

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input Directory Path", required=True)
# parser.add_argument("-o", "--output", help="Ouput Directory Path", required=True)
args = parser.parse_args()


def iter_docs(author):
    author_attr = author.attrib
    doc_dict = author_attr.copy()
    #    print(doc_dict)
    doc_dict['text'] = [' '.join([doc.text for doc in author.iter('document')])]
    count = 0
    countRT = 0
    for doc in author.iter('document'):
        if doc.text.endswith("#URL#"):
            count += 1
        if doc.text.startswith("RT"):
            countRT += 1
    doc_dict['endURL'] = count
    doc_dict['startRT'] = countRT
    return doc_dict


def create_data_frame(input_folder):
    os.chdir(input_folder)
    all_xml_files = glob.glob("*.xml")

    temp_list_of_DataFrames = []
    text_Data = pd.DataFrame()
    count_dict = {}
    for file in all_xml_files:
        etree = ET.parse(file)
        doc_df = pd.DataFrame(iter_docs(etree.getroot()))
        temp_list_of_DataFrames.append(doc_df)
    text_Data = pd.concat(temp_list_of_DataFrames, axis=0)
    text_Data = text_Data.drop(['text'], axis=1)
    return text_Data


def main():
    owd = os.getcwd()
    input = args.input + "/en"
    truth_data = pd.read_csv('results/en/truth.txt', sep=':::', names=['author_id', 'type'])
    predictions = create_data_frame(input)
    predictions.to_csv("counts.csv")
    predictionary = {}
    goldictionary = {}
    for index, row in predictions.iterrows():
        predictionary[row['id']] = int(row['type'])
    for index, row in truth_data.iterrows():
        goldictionary[row['author_id']] = row['type']

    predicts = []
    gold = []

    for key, value in predictionary.items():
        predicts.append(value)
        gold.append(goldictionary[key])

    print(confusion_matrix(gold, predicts))
    print(classification_report(gold, predicts))

    # Once again for ES

    os.chdir(owd)
    input = args.input + "/es"
    truth_data2 = pd.read_csv('results/es/truth.txt', sep=':::', names=['author_id', 'type'])
    predictions2 = create_data_frame(input)
    predictionary2 = {}
    goldictionary2 = {}
    for index, row in predictions2.iterrows():
        predictionary2[row['id']] = int(row['type'])
    for index, row in truth_data2.iterrows():
        goldictionary2[row['author_id']] = row['type']

    predicts2 = []
    gold2 = []

    for key, value in predictionary2.items():
        predicts2.append(value)
        gold2.append(goldictionary2[key])

    print(confusion_matrix(gold2, predicts2))
    print(classification_report(gold2, predicts2))


if __name__ == "__main__":
    main()

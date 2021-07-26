import argparse
import os

from simpletransformers.classification import ClassificationModel

from utils import read_data_into_DFs, data_for_bert, write_xml

"""
Write predictions from saved RoBERTa model to XMLs

Takes PATH to input folder and PATH to output location as arguments.

"""
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", required=True)
parser.add_argument("-o", "--output", required=True)

args = parser.parse_args()


def main():
    cwd = os.getcwd()
    DFs = read_data_into_DFs(args.input)
    os.chdir(cwd)

    # English

    text_for_bert_en = data_for_bert(DFs[0], train=False)
    model_en = ClassificationModel(
        "roberta",
        "models/bert/en/best_model",
        use_cuda=False)

    predictions_en, model_outputs = model_en.predict(text_for_bert_en)

    for i, prediction in enumerate(predictions_en):
        id = DFs[0].iloc[i]['authorID']
        write_xml(id, 'en', prediction, f"results/en/{id}.xml")

    # Spanish

    text_for_bert_es = data_for_bert(DFs[1], train=False)
    model_en = ClassificationModel(
        "roberta",
        "models/bert/es/best_model",
        use_cuda=False)

    predictions_es, model_outputs = model_en.predict(text_for_bert_en)

    for i, prediction in enumerate(predictions_en):
        id = DFs[1].iloc[i]['authorID']
        write_xml(id, 'es', prediction, f"results/es/{id}.xml")


if __name__ == "__main__":
    main()

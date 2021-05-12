import re
import string
import pandas as pd
import utils
from sklearn.svm import LinearSVC
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input Directory Path", required=True)
#parser.add_argument("-o", "--output", help="Ouput Directory Path", required=True)
args = parser.parse_args()


def build_fit_predict(type, params, xy):

    if type == 'LinearSVC':
        model = LinearSVC(**params)


    model.fit(xy[0], xy[2])

    return model.predict(xy[1])
def main():

    DFs = utils.read_data_into_DFs(args.input)

    #EN dataset
    vec_params_en = {
        "word_range": (1, 2),
        "word_min": 0.05,
        "word_max": 0.85,
        "char_range": (1, 6),
        "char_min": 0.001,
        "char_max": 1.0}

    xy_en = utils.create_xy(DFs[0], vec_params_en, concat=True)

    params_en = {
        "C": 22458.78034,
        "class_weight": "balanced",
        "intercept_scaling": 0.877,
        "loss": "hinge",
        "max_iter": 2000,
        "random_state": 42,
        "tol": 0.00014376187,
    }
    prediction_en = build_fit_predict('LinearSVC', params_en, xy_en)

    utils.get_results(prediction_en, xy_en[3], "English")

    #ES dataset
    vec_params_es = {
        "word_range": (1, 2),
        "word_min": 0.001,
        "word_max": 1.0,
        "char_range": (2, 6),
        "char_min": 0.001,
        "char_max": 0.75}

    xy_es = utils.create_xy(DFs[0], vec_params_es, concat=True)

    params_es = {
        "C": 8701.12835,
            "class_weight": None,
            "intercept_scaling": 9.095745,
            "loss": "squared_hinge",
            "max_iter": 2000,
            "random_state": 42,
            "tol": 0.0000017355634,}

    prediction_es = build_fit_predict('LinearSVC', params_es, xy_es)

    utils.get_results(prediction_es, xy_es[3], "Spanish")


if __name__ == "__main__":
    main()
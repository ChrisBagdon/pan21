import argparse
import csv
import logging
import os

import wandb
from simpletransformers.classification import ClassificationModel, ClassificationArgs
from sklearn.metrics import accuracy_score

from utils import read_data_into_DFs, data_for_bert

"""
Script to create RoBERTa/SpanBERTa model and records results.

Takes PATH to training data as argument
"""
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", required=True)
args = parser.parse_args()

# Create DFs
cwd = os.getcwd()
folder = args.input
DFs = read_data_into_DFs(folder)
train_df1, eval_df1 = data_for_bert(DFs[1])
os.chdir(cwd)

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# Set model arguments
model_args = ClassificationArgs()
model_args.learning_rate = 0.0000286
model_args.num_train_epochs = 1
model_args.wandb_project = "Bert Models Sp"
model_args.output_dir = "models/bert/es"
model_args.best_model_dir = "models/bert/es/best_model"
model_args.reprocess_input_data = True
model_args.overwrite_output_dir = True
model_args.evaluate_during_training = True
model_args.manual_seed = 4
model_args.use_multiprocessing = True
model_args.train_batch_size = 8
model_args.eval_batch_size = 4
model_args.labels_list = [0, 1]
"""model_args.use_early_stopping = False
model_args.early_stopping_delta = 0.001
model_args.early_stopping_metric = "mcc"
model_args.early_stopping_metric_minimize = False
model_args.early_stopping_patience = 5"""
model_args.evaluate_during_training_steps = 1000
model_args.evaluate_during_training_verbose = True
model_args.sliding_window = True
model_args.train_custom_parameters_only = False
model_args.gradient_accumulation_steps = 2
model_args.reprocess_input_data = True
model_args.no_save = False
model_args.no_cache = True
model_args.save_best_model = True
model_args.special_tokens_list = ["#EMOJI#", "#HASHTAG#", "#USER#", "#URL#"]
model_args.weight_decay = 0.1
wandb.init()

# Create a TransformerModel
model = ClassificationModel(
    "roberta",
    "skimai/spanberta-base-cased",
    # "roberta-base",
    use_cuda=True,
    args=model_args, )

model.train_model(train_df1, eval_df=eval_df1, accuracy=lambda truth, predictions: accuracy_score(
    truth, [round(p) for p in predictions]), )

model.eval_model(eval_df1)

wandb.join()

with open("es_runs.csv", "a", newline='') as file:
    writer = csv.writer(file, delimiter=",")

    writer.writerow([model.results["accuracy"], model.results["mcc"], str(model_args.learning_rate),
                     str(model_args.num_train_epochs)])

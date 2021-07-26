import logging

import wandb
from simpletransformers.classification import ClassificationModel, ClassificationArgs
from sklearn.metrics import accuracy_score

from utils import read_data_into_DFs, data_for_bert

"""
#https://simpletransformers.ai/docs/tips-and-tricks/

Simple Script for running hyper parameter sweeps via Wandb

"""

folder = "../pan21-author-profiling-training-2021-03-14"
DFs = read_data_into_DFs(folder)
train_df1, eval_df1 = data_for_bert(DFs[1])

sweep_config = {
    "method": "bayes",  # grid, random
    "metric": {"name": "train_loss", "goal": "minimize"},
    "parameters": {
        "num_train_epochs": {"values": [2, 3, 5]},
        "learning_rate": {"min": 1e-5, "max": 3e-5},
        "train_batch_size": {"values": [8, 16]}
    },
    "early_terminate": {"type": "hyperband", "min_iter": 6, },
}

sweep_id = wandb.sweep(sweep_config, project="Simple Sweep Spanish")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("transformers")
logger.setLevel(logging.WARNING)

model_args = ClassificationArgs()
model_args.num_train_epochs = 10
model_args.reprocess_input_data = True
model_args.overwrite_output_dir = True
model_args.evaluate_during_training = True
model_args.manual_seed = 4
model_args.use_multiprocessing = True
model_args.train_batch_size = 16
model_args.eval_batch_size = 4
model_args.labels_list = [0, 1]
model_args.wandb_project = "Simple Sweep Spanish "
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
model_args.no_save = True
model_args.no_cache = False
model_args.learning_rate = 4e-4
model_args.output_dir = "spanish_output"
model_args.best_model_dir = "spanish_output/best_model"
model_args.special_tokens_list = ["#EMOJI#", "#HASHTAG#", "#USER#", "#URL#"]
model_args.weight_decay = 0.1


def train():
    wandb.init()

    model = ClassificationModel(
        "roberta",
        "skimai/spanberta-base-cased",
        use_cuda=True,
        args=model_args,
        sweep_config=wandb.config, )

    model.train_model(train_df1, eval_df=eval_df1, accuracy=lambda truth, predictions: accuracy_score(
        truth, [round(p) for p in predictions]), )

    model.eval_model(eval_df1)

    wandb.join()


wandb.agent(sweep_id, train)

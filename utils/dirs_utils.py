"""
This module contains functions, which allow creation and management of the
model's run directory structure.
"""


import os
from os import path


__all__ = ["LOGS_ROOT", "CHECKPOINTS_DIR", "BEST_MODEL_DIR", "TRAIN_DIR",
           "VALIDATION_DIR", "does_run_exist", "create_run_dir"]


LOGS_ROOT = "logs"
CHECKPOINTS_DIR = "checkpoints"
BEST_MODEL_DIR = "best_model"
TRAIN_DIR = "train"
VAL_DIR = "val"


def create_run_dir(data_root, run_name):
    """
    Lay out run directory structure and return run's root. "val" is short for
    "validation".
    """
    run_root = path.join(path.join(data_root, LOGS_ROOT), run_name)
    os.mkdir(run_root)
    checkpoints_root = path.join(run_root, CHECKPOINTS_DIR)
    best_model_root = path.join(run_root, BEST_MODEL_DIR)
    train_root = path.join(run_root, TRAIN_DIR)
    val_root = path.join(run_root, VAL_DIR)
    for root in [checkpoints_root, best_model_root, train_root, val_root]:
        os.mkdir(root)
    return checkpoints_root, best_model_root, train_root, val_root


def initialize_run_dirs(data_root, run_name):
    """ Initialize all run directories, if it is possible to do so. """
    if path.isdir(path.join(path.join(data_root, LOGS_ROOT), run_name)):
        raise ValueError("Run with this name already exists.")
    roots = create_run_dir(data_root, run_name)
    return roots

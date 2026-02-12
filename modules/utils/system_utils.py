from datetime import datetime
from errno import EEXIST
from os import makedirs, path
import os
from shutil import copyfile, copytree, ignore_patterns
import cv2
def check_exists(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path, exist_ok=True)
        except OSError as e:
            if e.errno != EEXIST:
                raise
def copy_files(src_dir, dst_dir, *ignores):
    copytree(src_dir, dst_dir, ignore=ignore_patterns(*ignores))
def get_timestamp():
    return datetime.now().strftime(r"%y%m%d_%H%M%S")
def make_source_code_snapshot(log_dir):
    copy_files(
        ".",
        f"{log_dir}",
        ###### ignore files and directories ######
        "extensions",
        "assets",
        "output",
        "outputs",
        "__pycache__",
        "wandb",
        "Datasets",
        "data",
        "ckpts",
        "tmp",
        "ckpt",
        "logs",
        "scans",
        # ".vscode",
        "*.so",
        "*.a",
        ".ipynb_checkpoints",
        "build",
        "bin",
        "*.ply",
        "eigen",
        "pybind11",
        "*.npy",
        "*.pth",
        ".git",
        "debug",
        "assets",
        ".ipynb_checkpoints",
        ".md",
        ".gitignore",
        ".gitmodules",
        ".yml",
    )
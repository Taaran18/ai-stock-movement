import os, json, yaml
from pathlib import Path
from datetime import datetime


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def utcnow_str():
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

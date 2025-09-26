from pathlib import Path
import pickle
import os


def load_results(path):
    p = Path(path)
    if p.exists():
        with p.open("rb") as f:
            return pickle.load(f)
    return {}


def save_results(models_dic, path):
    tmp = str(path) + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(models_dic, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def merge_and_save(new_results: dict, path: str):
    models_dic = load_results(path)
    for k in new_results.keys():
        if k in models_dic:
            print(f"[warn] overwriting existing key: {k}")
    models_dic.update(new_results)
    save_results(models_dic, path)

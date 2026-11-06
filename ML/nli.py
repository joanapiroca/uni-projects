#!/usr/bin/env python3
import argparse
import lzma
import pickle
import os
from typing import Optional


import sklearn.feature_extraction.text
import sklearn.pipeline
import sklearn.metrics

import sklearn.svm

import numpy as np
import numpy.typing as npt

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="nli_competition.model", type=str, help="Model path")


class Dataset:
    CLASSES = ["ARA", "DEU", "FRA", "HIN", "ITA", "JPN", "KOR", "SPA", "TEL", "TUR", "ZHO"]

    def __init__(self, name="nli_dataset.train.txt"):
        if not os.path.exists(name):
            raise RuntimeError("The {} was not found, please download it from ReCodEx".format(name))

        # Load the dataset and split it into `data` and `target`.
        self.data, self.prompts, self.levels, self.target = [], [], [], []
        with open(name, "r", encoding="utf-8") as dataset_file:
            for line in dataset_file:
                target, prompt, level, text = line.rstrip("\n").split("\t")
                self.data.append(text)
                self.prompts.append(prompt)
                self.levels.append(level)
                self.target.append(-1 if not target else self.CLASSES.index(target))
        self.target = np.array(self.target, np.int32)


def main(args: argparse.Namespace) -> Optional[npt.ArrayLike]:
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()
        train_data, train_target = train.data, train.target


        # TODO: Train a model on the given dataset and store it in `model`.
        tf1 = sklearn.feature_extraction.text.TfidfVectorizer(ngram_range = (1,2), binary = True, use_idf = False, analyzer = 'word')
        clf = sklearn.svm.LinearSVC()
        pipe  = sklearn.pipeline.Pipeline(steps=[("tfidf", tf1), ("svc", clf)])
        model = pipe.fit(train_data, train_target)
        print(sklearn.metrics.accuracy_score(train_target, model.predict(train_data)))

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Use the model and return test set predictions.
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        # TODO: Generate `predictions` with the test set predictions, either
        # as a Python list or a NumPy array.
        predictions = model.predict(test.data)

        return predictions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
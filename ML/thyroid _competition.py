#!/usr/bin/env python3
import argparse
import lzma
import os
import pickle
import sys
from typing import Optional
import urllib.request

import numpy as np
import numpy.typing as npt
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="thyroid_competition.model", type=str, help="Model path")


class Dataset:
    """Thyroid Dataset.
    The dataset contains real medical data related to thyroid gland function,
    classified either as normal or irregular (i.e., some thyroid disease).
    The data consists of the following features in this order:
    - 15 binary features
    - 6 real-valued features
    The target variable is binary, with 1 denoting a thyroid disease and
    0 normal function.
    """
    def __init__(self,
                 name="thyroid_competition.train.npz",
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2223/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename=name)

        # Load the dataset and return the data and targets.
        dataset = np.load(name)
        for key, value in dataset.items():
            setattr(self, key, value)


def main(args: argparse.Namespace) -> Optional[npt.ArrayLike]:
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()
        X = train.data
        t = train.target
        
        from sklearn.model_selection import train_test_split
        X_train, X_test, t_train, t_test = train_test_split(X, t, test_size=5, random_state = args.seed)
        n_train = np.shape(X_train)[0]
        m_train = np.shape(X_train)[1]
        n_test = np.shape(X_test)[0]
        m_test = np.shape(X_test)[1]

        #Find categorical or real-calued columns in train and test data
        col_ints_train = []
    
        for j in range(m_train):
            if all(X_train[i,j].is_integer() for i in range(n_train)):
                col_ints_train.append(j)

        from sklearn.pipeline import Pipeline
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.preprocessing import StandardScaler
        from sklearn.preprocessing import PolynomialFeatures

        oh_transformer = OneHotEncoder(sparse = False, handle_unknown = "ignore") 
        col_transformer = ColumnTransformer([("cat", oh_transformer, col_ints_train)], remainder = StandardScaler())
    
        pol_transformer = PolynomialFeatures(2, include_bias=False)

        from sklearn.pipeline import Pipeline
        pipe = Pipeline([('coltrans', col_transformer), ('polynomial', pol_transformer)])      


        # TODO: Train a model on the given dataset and store it in `model`.
        log_reg = sklearn.linear_model.LogisticRegression()
        model = Pipeline([('pre-processor',pipe),('estimator',log_reg)])
        model.fit(X_train,t_train)

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Use the model and return test set predictions, either as a Python list or a NumPy array.
        test = Dataset(args.predict)
        X = test.data

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        # TODO: Generate `predictions` with the test set predictions.
        predictions = model.predict(X)
        print(predictions)

        return predictions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
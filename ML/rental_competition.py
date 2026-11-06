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
parser.add_argument("--model_path", default="rental_competition.model", type=str, help="Model path")


class Dataset:
    """Rental Dataset.
    The dataset instances consist of the following 12 features:
    - 0: season (1: winter, 2: spring, 3: summer, 4: autumn)
    - 1year (0: 2011, 1: 2012)
    - 2month (1-12)
    - 3hour (0-23)
    - 4holiday (binary indicator)
    - 5day of week (0: Sun, 1: Mon, ..., 6: Sat)
    - 6working day (binary indicator; a day is neither weekend nor holiday)
    - 7weather (1: clear, 2: mist, 3: light rain, 4: heavy rain)
    - 8temperature (normalized so that -8 Celsius is 0 and 39 Celsius is 1)
    - 9feeling temperature (normalized so that -16 Celsius is 0 and 50 Celsius is 1)
    - 10relative humidity (0-1 range)
    - 11windspeed (normalized to 0-1 range)
    The target variable is the number of rented bikes in the given hour.
    """
    def __init__(self,
                 name="rental_competition.train.npz",
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2223/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename=name)

        # Load the dataset and return the data and targets.
        dataset = np.load(name)
        for key, value in dataset.items():
            setattr(self, key, value)


from sklearn.pipeline import Pipeline

def raw_data(X) -> sklearn.pipeline.Pipeline :
    n = np.shape(X)[0]
    m = np.shape(X)[1]

    col_ints_train = []    
    for j in range(m):
        if all(X[i,j].is_integer() for i in range(n)):
            col_ints_train.append(j)

    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.preprocessing import StandardScaler

    oh_transformer = sklearn.preprocessing.OneHotEncoder(sparse = False, handle_unknown = "ignore") 
    col_transformer = ColumnTransformer([("cat", oh_transformer, col_ints_train)], remainder = StandardScaler())
    
    from sklearn.preprocessing import PolynomialFeatures
    pol_transformer = sklearn.preprocessing.PolynomialFeatures(2, include_bias=False)

    from sklearn.pipeline import Pipeline
    pipe = Pipeline([('coltrans', col_transformer), ('polynomial', pol_transformer)])      

    #train_data  = pipe.transform(X)
    #test_data = pipe.transform(X)
    return pipe


from sklearn.model_selection import train_test_split

def main(args: argparse.Namespace) -> Optional[npt.ArrayLike]:
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()
        X = train.data
        t = train.target

        X_train, X_test, t_train, t_test = train_test_split(X, t, test_size=5, random_state = args.seed)
        n_train = np.shape(X_train)[0]
        m_train = np.shape(X_train)[1]
        n_test = np.shape(X_test)[0]
        m_test = np.shape(X_test)[1]

        # TODO: Train a model on the given dataset and store it in `model`. 
        
        pipe= raw_data(X_train)
        lin_regression = sklearn.linear_model.SGDRegressor(max_iter=1000)
        model = Pipeline([('pre-processor',pipe),('estimator',lin_regression)])
        model.fit(X_train,t_train)
        
        """"
        trans_data = model.transform(X)
        regres = sklearn.linear_model.LinearRegression().fit(trans_data,t)
        prediction = regres.predict(trans_data)

        from sklearn.metrics import mean_squared_error
        rmse = mean_squared_error(t, prediction, squared=False)
        print(rmse)
        """

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


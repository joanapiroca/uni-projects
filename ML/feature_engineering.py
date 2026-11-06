#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.compose
import sklearn.datasets
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--dataset", default="diabetes", type=str, help="Standard sklearn dataset to load")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    dataset = getattr(sklearn.datasets, "load_{}".format(args.dataset))()
    X = dataset.data
    t = dataset.target

    # TODO: Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.
    from sklearn.model_selection import train_test_split
    X_train, X_test, t_train, t_test = train_test_split(X, t, test_size=args.test_size, random_state=args.seed)
    n_train = np.shape(X_train)[0]
    m_train = np.shape(X_train)[1]
    n_test = np.shape(X_test)[0]
    m_test = np.shape(X_test)[1]
    #print((n_train,m_train))


    # TODO: Process the input columns in the following way:
    #
    # - if a column has only integer values, consider it a categorical column
    #   (days in a week, dog breed, ...; in general, integer values can also
    #   represent numerical non-categorical values, but we use this assumption
    #   for the sake of exercise). Encode the values with one-hot encoding
    #   using `sklearn.preprocessing.OneHotEncoder` (note that its output is by
    #   default sparse, you can use `sparse=False` to generate dense output;
    #   also use `handle_unknown="ignore"` to ignore missing values in test set).
    #
    # - for the rest of the columns, normalize their values so that they
    #   have mean 0 and variance 1; use `sklearn.preprocessing.StandardScaler`.
    #
    # In the output, first there should be all the one-hot categorical features,
    # and then the real-valued features. To process different dataset columns
    # differently, you can use `sklearn.compose.ColumnTransformer`.
    
    #Find ints and floats columns in train and test data
    col_ints_train = []
    
    for j in range(m_train):
        if all(X_train[i,j].is_integer() for i in range(n_train)):
            col_ints_train.append(j)
            
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.preprocessing import StandardScaler
   
    oh_transformer = sklearn.preprocessing.OneHotEncoder(sparse = False, handle_unknown = "ignore") 
    scaler_transformer = StandardScaler() 

    col_transformer = ColumnTransformer([("ints", oh_transformer, col_ints_train)], remainder=scaler_transformer)

    # TODO: To the current features, append polynomial features of order 2.
    # If the input values are `[a, b, c, d]`, you should append
    # `[a^2, ab, ac, ad, b^2, bc, bd, c^2, cd, d^2]`. You can generate such polynomial
    # features either manually, or you can generate them with
    # `sklearn.preprocessing.PolynomialFeatures(2, include_bias=False)`.
    from sklearn.preprocessing import PolynomialFeatures
    pol_transformer = sklearn.preprocessing.PolynomialFeatures(2, include_bias=False)


    # TODO: You can wrap all the feature processing steps into one transformer
    # by using `sklearn.pipeline.Pipeline`. Although not strictly needed, it is
    # usually comfortable.
    from sklearn.pipeline import Pipeline
    pipe = Pipeline([('coltrans', col_transformer), ('polynomial', pol_transformer)])

    # TODO: Fit the feature preprocessing steps (the composed pipeline with all of
    # them; or the individual steps, if you prefer) on the training data (using `fit`).
    # Then transform the training data into `train_data` (with a `transform` call;
    # however, you can combine the two methods into a single `fit_transform` call).
    # Finally, transform testing data to `test_data`.
    pipe.fit(X_train)
    train_data  = pipe.transform(X_train)
    test_data = pipe.transform(X_test)

    

    return train_data[:5], test_data[:5]


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    train_data, test_data = main(args)
    for dataset in [train_data, test_data]:
        for line in range(min(dataset.shape[0], 5)):
            print(" ".join("{:.4g}".format(dataset[line, column]) for column in range(min(dataset.shape[1], 140))))
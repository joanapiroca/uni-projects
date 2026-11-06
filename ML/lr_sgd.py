#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size")
parser.add_argument("--data_size", default=100, type=int, help="Data size")
parser.add_argument("--epochs", default=50, type=int, help="Number of SGD training epochs")
parser.add_argument("--l2", default=0.0, type=float, help="L2 regularization strength")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
parser.add_argument("--plot", default=False, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> tuple[list[float], float, float]:
    # Create a random generator with a given seed
    generator = np.random.RandomState(args.seed)

    # Generate an artificial regression dataset
    X, t = sklearn.datasets.make_regression(n_samples=args.data_size, random_state=args.seed)
    n = np.shape(X)[0]
    m = np.shape(X)[1]

    # TODO: Append a constant feature with value 1 to the end of every input data
    col_add = np.ones((n, 1))
    X = np.append(X, col_add , axis = 1)

    # TODO: Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.
    from sklearn.model_selection import train_test_split
    X_train, X_test, t_train, t_test = train_test_split(X, t, test_size=args.test_size, random_state=args.seed)

    # Generate initial linear regression weights
    weights = generator.uniform(size=X_train.shape[1], low=-0.1, high=0.1)

    from sklearn.metrics import mean_squared_error
    b = args.batch_size
    train_rmses, test_rmses = [], []
    for epoch in range(args.epochs):
        permutation = generator.permutation(X_train.shape[0])

        # TODO: Process the data in the order of `permutation`. For every
        # `args.batch_size` of them, average their gradient, and update the weights.
        # A gradient for example `(x_i, t_i)` is `(x_i^T weights - t_i) * x_i`,
        # and the SGD update is
        #   weights = weights - args.learning_rate * (gradient + args.l2 * weights)`.
        # You can assume that `args.batch_size` exactly divides `train_data.shape[0]`.

        X_perm = []
        X_perm = X_train[permutation]

        t_perm = []
        t_perm = t_train[permutation]

        for i in range(int(np.shape(X_train)[0]/b)):            
            gradient = 0
            for j in range(int(b)):
                gradient += (np.subtract(np.dot(X_perm[b*i+j], weights), t_perm[b*i+j]))*X_perm[b*i+j]
            gradient = gradient/b
            weights -= args.learning_rate * (gradient + args.l2 * weights)
            
        # TODO: Append current RMSE on train/test to `train_rmses`/`test_rmses`.
        RMSE_train = mean_squared_error(t_train, np.dot(X_train,weights), squared=False)
        RMSE_test = mean_squared_error(t_test, np.dot(X_test,weights), squared = False)
        train_rmses.append(RMSE_train)
        test_rmses.append(RMSE_test)

    # TODO: Compute into `explicit_rmse` test data RMSE when fitting
    # `sklearn.linear_model.LinearRegression` on `train_data` (ignoring `args.l2`).
    model = sklearn.linear_model.LinearRegression().fit(X_train, t_train)
    explicit_rmse = mean_squared_error(t_test, model.predict(X_test), squared=False)

    if args.plot:
        import matplotlib.pyplot as plt
        plt.plot(train_rmses, label="Train")
        plt.plot(test_rmses, label="Test")
        plt.xlabel("Epochs")
        plt.ylabel("RMSE")
        plt.legend()
        plt.show() if args.plot is True else plt.savefig(args.plot, transparent=True, bbox_inches="tight")

    return weights, test_rmses[-1], explicit_rmse


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    weights, sgd_rmse, explicit_rmse = main(args)
    print("Test RMSE: SGD {:.2f}, explicit {:.2f}".format(sgd_rmse, explicit_rmse))
    print("Learned weights:", *("{:.2f}".format(weight) for weight in weights[:12]), "...")
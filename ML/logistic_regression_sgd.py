#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size")
parser.add_argument("--data_size", default=100, type=int, help="Data size")
parser.add_argument("--epochs", default=50, type=int, help="Number of SGD training epochs")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
parser.add_argument("--plot", default=False, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
# If you add more arguments, ReCodEx will keep them with your default values.

def sigmoid(x):
    return np.divide(1,1+np.exp(-x))

def main(args: argparse.Namespace) -> tuple[np.ndarray, list[tuple[float, float]]]:
    # Create a random generator with a given seed
    generator = np.random.RandomState(args.seed)

    # Generate an artificial classification dataset
    data, target = sklearn.datasets.make_classification(
        n_samples=args.data_size, n_features=2, n_informative=2, n_redundant=0, random_state=args.seed)

    # TODO: Append a constant feature with value 1 to the end of every input data
    data = np.append(data, np.ones((np.shape(data)[0], 1)), axis=1)

    # TODO: Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.
    X_train, X_test, t_train, t_test = sklearn.model_selection.train_test_split(data, target, test_size=args.test_size, random_state = args.seed)
    n_train = np.shape(X_train)[0]
    m_train = np.shape(X_train)[1]
    n_test = np.shape(X_test)[0]
    m_test = np.shape(X_test)[1]

    # Generate initial logistic regression weights
    weights = generator.uniform(size=X_train.shape[1], low=-0.1, high=0.1)
    b = args.batch_size
    alpha = args.learning_rate


    for epoch in range(args.epochs):
        permutation = generator.permutation(X_train.shape[0])

        # TODO: Process the data in the order of `permutation`. For every
        # `args.batch_size` of them, average their gradient, and update the weights.
        # You can assume that `args.batch_size` exactly divides `train_data.shape[0]`.
        X_perm = []
        X_perm = X_train[permutation]

        t_perm = []
        t_perm = t_train[permutation]

        for i in range(int(n_train/b)):            
            gradient = 0
            for j in range(int(b)):
                y = sigmoid(np.dot(X_perm[b*i+j], weights))
                gradient += (y - t_perm[b*i+j])*X_perm[b*i+j]
            gradient = gradient/b
            weights -= alpha * gradient

        # TODO: After the SGD epoch, measure the average loss and accuracy for both the
        # train set and the test set. The loss is the average MLE loss (i.e., the
        # negative log likelihood, or crossentropy loss, or KL loss) per example.        

        #Predictions y 
        y_train = [sigmoid(np.dot(X_train[rows], weights)) for rows in range(n_train)]
        y_test = [sigmoid(np.dot(X_test[rows], weights)) for rows in range(n_test)]
        
        #Loss
        train_loss = sklearn.metrics.log_loss(t_train, y_train)
        test_loss = sklearn.metrics.log_loss(t_test, y_test)
        #train_loss = (1/n_train)*np.sum(-np.log(y_train))
        #test_loss = (1/n_test)*np.sum(-np.log(y_test))

        y_train_pred = []
        y_test_pred = []

        for rows in range(n_train):
            if y_train[rows] >= 0.5: y_train_pred.append(1)
            else: y_train_pred.append(0)
        for rows in range(n_test):
            if y_test[rows] >= 0.5: y_test_pred.append(1)
            else: y_test_pred.append(0)

        #Accuracy 
        corrects_train = 0
        for i in range(n_train):
            if (y_train_pred[i] == t_train[i]) :   #This will be always false because the sigmoidal function doesn't have either 1 or zero in it's image!!!!!!
                corrects_train +=1

        corrects_test = 0
        for i in range(n_test):
            if (y_test_pred[i] == t_test[i]) :                
                corrects_test +=1

        train_accuracy = np.divide(corrects_train, n_train)
        test_accuracy = np.divide(corrects_test, n_test)       


        print("After epoch {}: train loss {:.4f} acc {:.1f}%, test loss {:.4f} acc {:.1f}%".format(
            epoch + 1, train_loss, 100 * train_accuracy, test_loss, 100 * test_accuracy))

        if args.plot:
            import matplotlib.pyplot as plt
            if args.plot is not True:
                plt.gcf().get_axes() or plt.figure(figsize=(6.4*3, 4.8*(args.epochs+2)//3))
                plt.subplot(3, (args.epochs+2)//3, 1 + epoch)
            xs = np.linspace(np.min(data[:, 0]), np.max(data[:, 0]), 50)
            ys = np.linspace(np.min(data[:, 1]), np.max(data[:, 1]), 50)
            predictions = [[1 / (1 + np.exp(-([x, y, 1] @ weights))) for x in xs] for y in ys]
            plt.contourf(xs, ys, predictions, levels=21, cmap="RdBu", alpha=0.7)
            plt.contour(xs, ys, predictions, levels=[0.25, 0.5, 0.75], colors="k")
            plt.scatter(X_train[:, 0], X_train[:, 1], c=t_train, label="train", marker="P", cmap="RdBu")
            plt.scatter(X_test[:, 0], X_test[:, 1], c=t_test, label="test", cmap="RdBu")
            plt.legend(loc="upper right")
            plt.show() if args.plot is True else plt.savefig(args.plot, transparent=True, bbox_inches="tight")

    return weights, [(train_loss, 100 * train_accuracy), (test_loss, 100 * test_accuracy)]


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    weights, metrics = main(args)
    print("Learned weights", *("{:.2f}".format(weight) for weight in weights))
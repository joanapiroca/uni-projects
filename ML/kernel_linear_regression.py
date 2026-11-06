#!/usr/bin/env python3
# d9034c78-c8b2-4820-8671-0e919beeaaf3
# 10d1c6c4-6216-4721-b23c-d0de3e0fb138
# c8cb695-9dc2-465b-b4ea-1637eb270929
import argparse

import numpy as np
import sklearn.metrics

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=1, type=int, help="Batch size")
parser.add_argument("--data_size", default=50, type=int, help="Data size")
parser.add_argument("--epochs", default=200, type=int, help="Number of SGD training epochs")
parser.add_argument("--kernel", default="rbf", type=str, help="Kernel type (poly/rbf)")
parser.add_argument("--kernel_degree", default=3, type=int, help="Degree for poly kernel")
parser.add_argument("--kernel_gamma", default=1.0, type=float, help="Gamma for poly and rbf kernel")
parser.add_argument("--l2", default=0.0, type=float, help="L2 regularization weight")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
parser.add_argument("--plot", default=False, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# If you add more arguments, ReCodEx will keep them with your default values.

def pol_kernel(x,z,gamma,d):
    return np.power(gamma*np.dot(x,z)+1,d)

def rbf_kernel(x,z,gamma):
    return np.exp(-gamma*(np.linalg.norm(x-z)**2))

def main(args: argparse.Namespace) -> tuple[np.ndarray, float, list[float], list[float]]:
    # Create a random generator with a given seed.
    generator = np.random.RandomState(args.seed)

    # Generate an artificial regression dataset.
    train_data = np.linspace(-1, 1, args.data_size)
    train_target = np.sin(5 * train_data) + generator.normal(scale=0.25, size=args.data_size) + 1
    train_data = np.expand_dims(train_data, axis=1)

    test_data = np.linspace(-1.2, 1.2, 2*args.data_size)
    test_target = np.sin(5 * test_data) + 1
    test_data = np.expand_dims(test_data, axis=1)

    # Initialize the parameters: the betas and the bias.
    betas = np.zeros(args.data_size)
    bias = 0

    gamma = args.kernel_gamma
    d = args.kernel_degree
    b = args.batch_size
    n_train = np.shape(train_data)[0] 
    n_test = np.shape(test_data)[0]
    lr = args.learning_rate
    
    Ker_train = np.zeros((n_train,n_train))
    for p in range(n_train) :
        for t in range(n_train) :
            if args.kernel == 'poly': Ker_train[p][t] = pol_kernel(train_data[p], train_data[t], gamma, d)
            if args.kernel == 'rbf' : Ker_train[p][t] = rbf_kernel(train_data[p], train_data[t], gamma)

    #I will also need the Ker of the test data for computing the rmse:
    Ker_test = np.zeros((n_test,n_test))
    for w in range(n_test):
        for q in range(n_train):
            if args.kernel == 'poly': Ker_test[w][q] = pol_kernel(test_data[w], train_data[q], gamma, d)
            if args.kernel == 'rbf' : Ker_test[w][q] = rbf_kernel(test_data[w], train_data[q], gamma)

    train_rmses, test_rmses = [],[]
    for epoch in range(args.epochs):
        permutation = generator.permutation(train_data.shape[0])

        for i in range(0, len(train_data), args.batch_size):
            perm_batch = permutation[i:(i+b)]
            target_batch = train_target[perm_batch]
            grad = np.zeros(b)

            grad = Ker_train[perm_batch]@betas + bias - target_batch
            bias -= (lr/b)*np.sum(grad)
            new_betas = betas
            new_betas[perm_batch] = betas[perm_batch] - (lr/b)*grad
            new_betas -= (args.l2*lr)*betas

            betas = new_betas

        #RMSE: I first need the predictions
        pred_train = Ker_train@betas+bias
        pred_test  = Ker_test@np.concatenate((betas,betas),axis=0) + bias    #BETAS HAVE DIMENSION N_TRAIN = ARGS.DATA_SIZE AND KER_TEST HAS DIMENSION N_TEST = 2*ARGS.DATA_SIZE!!!!!! Thwy don't match!!!!!!!!!!!!

        rmse_train = np.sqrt((1/n_train)*np.sum([(pred_train[i]-train_target[i])**2 for i in range(n_train)]))
        rmse_test = np.sqrt((1/n_test)*np.sum([(pred_test[i]-test_target[i])**2 for i in range(n_test)]))             


        # TODO: Append current RMSE on train/test data to `train_rmses`/`test_rmses`.
        train_rmses.append(rmse_train)
        test_rmses.append(rmse_test)

        if (epoch + 1) % 10 == 0:
            print("After epoch {}: train RMSE {:.2f}, test RMSE {:.2f}".format(
                epoch + 1, train_rmses[-1], test_rmses[-1]))

    if args.plot:
        import matplotlib.pyplot as plt
        # If you want the plotting to work (not required for ReCodEx), compute the `test_predictions`.
        test_predictions = ...

        plt.plot(train_data, train_target, "bo", label="Train target")
        plt.plot(test_data, test_target, "ro", label="Test target")
        plt.plot(test_data, test_predictions, "g-", label="Predictions")
        plt.legend()
        plt.show() if args.plot is True else plt.savefig(args.plot, transparent=True, bbox_inches="tight")

    return betas, bias, train_rmses, test_rmses


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    betas, bias, train_rmses, test_rmses = main(args)
    print("Learned betas", *("{:.2f}".format(beta) for beta in betas[:15]), "...")
    print("Learned bias", bias)
#!/usr/bin/env python3
# d9034c78-c8b2-4820-8671-0e919beeaaf3
# 10d1c6c4-6216-4721-b23c-d0de3e0fb138
# c8cb695-9dc2-465b-b4ea-1637eb270929
import argparse

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import math

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--C", default=1, type=float, help="Inverse regularization strength")
parser.add_argument("--data_size", default=200, type=int, help="Data size")
parser.add_argument("--kernel", default="poly", type=str, help="Kernel type (poly/rbf)")
parser.add_argument("--kernel_degree", default=1, type=int, help="Degree for poly kernel")
parser.add_argument("--kernel_gamma", default=1.0, type=float, help="Gamma for poly and rbf kernel")
parser.add_argument("--max_iterations", default=1000, type=int, help="Maximum number of iterations to perform")
parser.add_argument("--max_passes_without_as_changing", default=10, type=int, help="Stopping condition")
parser.add_argument("--plot", default=False, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
parser.add_argument("--tolerance", default=1e-7, type=float, help="Default tolerance for KKT conditions")
# If you add more arguments, ReCodEx will keep them with your default values.


def kernel(args: argparse.Namespace, x: np.ndarray, z: np.ndarray) -> np.ndarray:
    K = np.zeros((np.shape(x)[0],np.shape(z)[0]))
    gamma = args.kernel_gamma
    d = args.kernel_degree
    for i in range(np.shape(x)[0]) :
        for j in range(np.shape(z)[0]) :
            if args.kernel == 'poly': K[i][j] = np.power(gamma*np.dot(x[i],z[j])+1,d)
            if args.kernel == 'rbf' : K[i][j] = np.exp(-gamma*(np.linalg.norm(x[i]-z[j])**2))
    
    return K

def prediction(index,t,a,K,b,tol): #where index in the index of the x_i we're computing it's prediction
    pred = []
    for i in range(len(a)):
        if a[i] > tol:
            pred.append(a[i]*t[i]*K[index][i])
    return np.sum(pred)+b

def compute_L(alpha_i, alpha_j, y_i, y_j, C):
    if y_i != y_j:
        return max(0, alpha_j - alpha_i)
    else:
        return max(0, alpha_i + alpha_j - C)


def compute_H(alpha_i, alpha_j, y_i, y_j, C):
    if y_i != y_j:
        return min(C, C + alpha_j - alpha_i)
    else:
        return min(C, alpha_i + alpha_j)

def clip_alpha_j(alpha_j, H, L,tol):
    if alpha_j > H+tol:
        return H
    elif alpha_j < L-tol:
        return L
    else:
        return alpha_j

def calculate_alpha_j(alpha_j, E_i, E_j, eta, y_j):
    return alpha_j - ((y_j * (E_i - E_j)) / eta)


def calculate_alpha_i(alpha_i, y_i, y_j, alpha_j_old, alpha_j):
    return alpha_i + (y_i * y_j) * (alpha_j_old - alpha_j)


def compute_b(b_i, b_j, alpha_i, alpha_j, C,tol):
    if alpha_i > tol and alpha_i <= C-tol:
        return b_i
    elif alpha_j > tol and alpha_j <= C-tol:
        return b_j
    else:
        return ((b_i + b_j) / 2)

# We implement the SMO algorithm as a separate method, so we can use
# it in the svm_multiclass assignment too.
def smo(
    args: argparse.Namespace,
    train_data: np.ndarray, train_target: np.ndarray,
    test_data: np.ndarray, test_target: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float, list[float], list[float]]:
    # Create initial weights.
    a, b = np.zeros(len(train_data)), 0
    alphas_old = np.zeros(len(train_data))
    generator = np.random.RandomState(args.seed)
    C = args.C
    tol = args.tolerance
    K = kernel(args,train_data,train_data)
    passes_without_as_changing = 0
    train_accs, test_accs = [], []
    for _ in range(args.max_iterations):
        as_changed = 0
        # Iterate through the data.
        for i, j in enumerate(generator.randint(len(a) - 1, size=len(a))):
            # We want j != i, so we "skip" over the value of i.
            j = j + (j >= i)

            # TODO: Check that a[i] fulfills the KKT conditions, using `args.tolerance` during comparisons.

            # If the conditions do not hold, then:
            # - compute the updated unclipped a_j^new.
            #
            #   If the second derivative of the loss with respect to a[j]
            #   is > -`args.tolerance`, do not update a[j] and continue
            #   with next i.

            # - clip the a_j^new to suitable [L, H].
            #
            #   If the clipped updated a_j^new differs from the original a[j]
            #   by less than `args.tolerance`, do not update a[j] and continue
            #   with next i.

            # - update a[j] to a_j^new, and compute the updated a[i] and b.
            #
            #   During the update of b, compare the a[i] and a[j] to zero by
            #   `> args.tolerance` and to C using `< args.C - args.tolerance`.

            # - increase `as_changed`.

            E_i = (prediction(i,train_target,a,K,b,tol)-train_target[i])
            y_i_E_i = train_target[i] * E_i
            

            if (((a[i] < (C - tol)) and (y_i_E_i < (-1*tol))) or ((a[i] > tol) and (y_i_E_i > tol))):
                
                E_j = (prediction(j,train_target,a,K,b,tol)-train_target[j])
                alphas_old[i], alphas_old[j] = a[i], a[j]
                second_derivative = 2*K[i][j] - K[i][i] - K[j][j]
                a_j_new = calculate_alpha_j(a[j],E_i,E_j,second_derivative,train_target[j])

                if(second_derivative > (-1*tol)): 
                    continue
                    
                L = compute_L(a[i], a[j], train_target[i], train_target[j], C)
                H = compute_H(a[i], a[j], train_target[i], train_target[j], C)
                a_j_new = clip_alpha_j(a_j_new,H,L,tol)

                if L == H:
                    continue
                if(abs(a_j_new-a[j]) < tol):
                    continue

                a[j] = a_j_new
                a[i] = calculate_alpha_i(a[i],train_target[i],train_target[j],alphas_old[j],a[j])
                b_j = b - E_j - train_target[i]*(a[i]-alphas_old[i])*K[i][j] - train_target[j]*(a[j] - alphas_old[j])*K[j][j]
                b_i = b - E_i - train_target[i]*(a[i]-alphas_old[i])*K[i][i] - train_target[j]*(a[j] - alphas_old[j])*K[j][i]
                b = compute_b(b_i,b_j,a[i],a[j],C,tol)
                as_changed += 1


        y_train = [prediction(k,train_target,a,K,b,tol) for k in range(len(train_data))]
        K_test = kernel(args,test_data,train_data)
        y_test = [prediction(k,train_target,a,K_test,b,tol) for k in range(len(test_data))]
        
        # TODO: After each iteration, measure the accuracy for both the
        # train set and the test set and append it to `train_accs` and `test_accs`.
        
        
        train_acc = 0
        for k in range(len(train_data)):
            if(np.sign(y_train[k]) == np.sign(train_target[k])): train_acc+=1
        #print(train_acc)
            
        test_acc = 0
        for k in range(len(test_data)):
            if(np.sign(y_test[k]) == np.sign(test_target[k])): test_acc+=1
        
        train_accs.append(train_acc/len(train_data))
        test_accs.append(test_acc/len(test_data))

        # Stop training if `args.max_passes_without_as_changing` passes were reached.
        passes_without_as_changing = 0 if as_changed else passes_without_as_changing + 1
        if passes_without_as_changing >= args.max_passes_without_as_changing:
            break

        if len(train_accs) % 100 == 0 and len(train_accs) < args.max_iterations:
            print("Iteration {}, train acc {:.1f}%, test acc {:.1f}%".format(
                len(train_accs), 100 * train_accs[-1], 100 * test_accs[-1]))

    # TODO: Create an array of support vectors (in the same order in which they appeared
    # in the training data; to avoid rounding errors, consider a training example
    # a support vector only if a_i > `args.tolerance`) and their weights (a_i * t_i).

    support_vectors, support_vector_weights = [],[]
    for k in range(len(train_data)):
        if a[k] > args.tolerance:
            support_vectors.append(train_data[k])
            support_vector_weights.append(a[k]*train_target[k])
            
    
    print("Done, iteration {}, support vectors {}, train acc {:.1f}%, test acc {:.1f}%".format(
        len(train_accs), len(support_vectors), 100 * train_accs[-1], 100 * test_accs[-1]))

    return support_vectors, support_vector_weights, b, train_accs, test_accs


def main(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, float, list[float], list[float]]:
    # Generate an artificial regression dataset, with +-1 as targets.
    data, target = sklearn.datasets.make_classification(
        n_samples=args.data_size, n_features=2, n_informative=2, n_redundant=0, random_state=args.seed)
    target = 2 * target - 1

    # Split the dataset into a train set and a test set.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    # Run the SMO algorithm.
    support_vectors, support_vector_weights, bias, train_accs, test_accs = smo(
        args, train_data, train_target, test_data, test_target)

    if args.plot:
        import matplotlib.pyplot as plt

        def plot(predict, support_vectors):
            xs = np.linspace(np.min(data[:, 0]), np.max(data[:, 0]), 50)
            ys = np.linspace(np.min(data[:, 1]), np.max(data[:, 1]), 50)
            predictions = [[predict(np.array([x, y])) for x in xs] for y in ys]
            test_mismatch = np.sign([predict(x) for x in test_data]) != test_target
            plt.figure()
            plt.contourf(xs, ys, predictions, levels=0, cmap="RdBu")
            plt.contour(xs, ys, predictions, levels=[-1, 0, 1], colors="k", zorder=1)
            plt.scatter(train_data[:, 0], train_data[:, 1], c=train_target, marker="o", label="Train", cmap="RdBu", zorder=2)
            plt.scatter(support_vectors[:, 0], support_vectors[:, 1], marker="o", s=90, label="Support Vectors", c="#0d0")
            plt.scatter(test_data[:, 0], test_data[:, 1], c=test_target, marker="*", label="Test", cmap="RdBu", zorder=2)
            plt.scatter(test_data[test_mismatch, 0], test_data[test_mismatch, 1], marker="*", s=130, label="Test Errors", c="#ff0")
            plt.legend(loc="upper center", ncol=4)

        # If you want plotting to work (not required for ReCodEx), you need to
        # define `predict_function` computing SVM value `y(x)` for the given x.
        def predict_function(x):
            return ...

        plot(predict_function, support_vectors)
        plt.show() if args.plot is True else plt.savefig(args.plot, transparent=True, bbox_inches="tight")

    return support_vectors, support_vector_weights, bias, train_accs, test_accs


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size")
parser.add_argument("--classes", default=10, type=int, help="Number of classes to use")
parser.add_argument("--epochs", default=10, type=int, help="Number of SGD training epochs")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=797, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> tuple[np.ndarray, list[tuple[float, float]]]:
    # Create a random generator with a given seed
    generator = np.random.RandomState(args.seed)

    # Use the digits dataset
    data, target = sklearn.datasets.load_digits(n_class=args.classes, return_X_y=True)

    # Append a constant feature with value 1 to the end of every input data.
    # Then we do not need to explicitly represent bias - it becomes the last weight.
    data = np.pad(data, [(0, 0), (0, 1)], constant_values=1)

    # Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.
    X_train, X_test, t_train, t_test = sklearn.model_selection.train_test_split(data, target, test_size=args.test_size, random_state = args.seed)
    n_train = np.shape(X_train)[0]
    m_train = np.shape(X_train)[1]
    n_test = np.shape(X_test)[0]
    m_test = np.shape(X_test)[1]

    # Generate initial model weights
    weights = generator.uniform(size=[X_train.shape[1], args.classes], low=-0.1, high=0.1)
    from sklearn.metrics import mean_squared_error
    from sklearn.preprocessing import OneHotEncoder

    b = args.batch_size
    alpha = args.learning_rate


    for epoch in range(args.epochs):
        permutation = generator.permutation(X_train.shape[0])

        # TODO: Process the data in the order of `permutation`. For every
        # `args.batch_size` of them, average their gradient, and update the weights.
        # You can assume that `args.batch_size` exactly divides `train_data.shape[0]`.
        #
        # Note that you need to be careful when computing softmax because the exponentiation
        # in softmax can easily overflow. To avoid it, you should use the fact that
        # $softmax(z) = softmax(z + any_constant)$ and compute $softmax(z) = softmax(z - maximum_of_z)$.
        # That way we only exponentiate non-positive values, and overflow does not occur.
        X_perm = []
        X_perm = X_train[permutation]
        t_perm = []
        t_perm = t_train[permutation]
        #onehotencouder de t
        t_oh = OneHotEncoder(sparse = False, handle_unknown = "ignore").fit_transform(t_perm.reshape((np.shape(t_perm)[0],1)))

        for i in range(int(n_train/b)):            
            gradient = 0
            for j in range(int(b)):
                #comprovar si xt*W és una columna sencera (vull això) o només un element(no vull)
                xw_no_norm = np.dot(X_perm[b*i+j],weights)
                xw = [xw_no_norm[k]-np.amax(xw_no_norm) for k in range(np.shape(xw_no_norm)[0])]
                aux = [np.exp(xw[k]) for k in range(np.shape(weights)[1])]
                y = [np.divide(aux[k], np.sum(aux)) for k in range(np.shape(weights)[1] )]
                
                gradient += np.outer((y - t_oh[b*i+j]), X_perm[b*i+j])   #amb la t de onehot encouder
            gradient = gradient/b
            weights -= alpha * gradient.T

        # TODO: After the SGD epoch, measure the average loss and accuracy for both the
        # train test and the test set. The loss is the average MLE loss (i.e., the
        # negative log likelihood, or cross-entropy loss, or KL loss) per example.
       

        #TRAIN PREDICTIONS
        xw_no = np.dot(X_train, weights)
        xw = [xw_no[k]-np.amax(xw_no[k]) for k in range(np.shape(xw_no)[0])]
        xw_exp = np.exp(xw)
        y_oh = np.empty(np.shape(xw))
        for i in range(np.shape(xw)[0]):
            for j in range(np.shape(xw)[1]):
                y_oh[i][j] = np.divide(xw_exp[i][j], np.sum(xw_exp[i]))

        y_train = np.empty(np.shape(t_train))
        for i in range(np.shape(y_oh)[0]):
            y_train[i] = int(np.argmax(y_oh[i]))

        
        #TEST PREDICTIONS
        xw_no = np.dot(X_test, weights)
        xw = [xw_no[k]-np.amax(xw_no[k]) for k in range(np.shape(xw_no)[0])]
        xw_exp = np.exp(xw)
        y_oh_test = np.empty(np.shape(xw))

        for i in range(np.shape(xw)[0]):
            for j in range(np.shape(xw)[1]):
                y_oh_test[i][j] = np.divide(xw_exp[i][j], np.sum(xw_exp[i]))

        y_test = [np.argmax(y_oh_test[i]) for i in range(np.shape(y_oh_test)[0])]



        #Loss
        train_loss = sklearn.metrics.log_loss(t_train, y_oh)
        test_loss = sklearn.metrics.log_loss(t_test, y_oh_test)
        #train_loss = (1/n_train)*np.sum([(1/m_train)*np.sum(-np.log(y_oh[i])) for i in range(n_train)])
        #test_loss = (1/n_test)*np.sum([(1/m_test)*np.sum(-np.log(y_test[i])) for i in range(n_test)])
       
        #Accuracy
        corrects_train = 0
        for i in range(n_train):
            if y_train[i] == t_train[i] :
                corrects_train +=1

        corrects_test = 0
        for i in range(n_test):
            if y_test[i] == t_test[i] :                
                corrects_test +=1

        train_accuracy = np.divide(corrects_train, n_train)
        test_accuracy = np.divide(corrects_test, n_test)       



        print("After epoch {}: train loss {:.4f} acc {:.1f}%, test loss {:.4f} acc {:.1f}%".format(
            epoch + 1, train_loss, 100 * train_accuracy, test_loss, 100 * test_accuracy))

    return weights, [(train_loss, 100 * train_accuracy), (test_loss, 100 * test_accuracy)]


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    weights, metrics = main(args)
    print("Learned weights:",
          *(" ".join([" "] + ["{:.2f}".format(w) for w in row[:10]] + ["..."]) for row in weights.T), sep="\n")
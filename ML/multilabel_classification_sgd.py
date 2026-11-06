#!/usr/bin/env python3

# Joana Pi Roca: d9034c78-c8b2-4820-8671-0e919beeaaf3
# Jorge Torró Fernández: 10d1c6c4-6216-4721-b23c-d0de3e0fb138
# Juan Cubo Bravo: 8c8cb695-9dc2-465b-b4ea-1637eb270929

import argparse

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size")
parser.add_argument("--classes", default=5, type=int, help="Number of classes to use")
parser.add_argument("--data_size", default=200, type=int, help="Data size")
parser.add_argument("--epochs", default=10, type=int, help="Number of SGD training epochs")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
# If you add more arguments, ReCodEx will keep them with your default values.

def sigmoid(z):
    return 1.0/(1 + np.exp(-z))

def values(predictions,targets):
    pred = []
    for i in range(predictions.shape[0]):
        predict = 1
        if(predictions[i] < 0.5):
            predict = 0
        pred.append(predict)      
    return pred

def main(args: argparse.Namespace) -> tuple[np.ndarray, list[tuple[float, float]]]:
    # Create a random generator with a given seed.
    generator = np.random.RandomState(args.seed)

    # Generate an artificial classification dataset.
    data, target_list = sklearn.datasets.make_multilabel_classification(
        n_samples=args.data_size, n_classes=args.classes, allow_unlabeled=False,
        return_indicator=False, random_state=args.seed)

    # TODO: The `target` is a list of classes for every input example. Convert
    # it to a dense representation (n-hot encoding) -- for each input example,
    # the target should be vector of `args.classes` binary indicators.
    target = sklearn.preprocessing.MultiLabelBinarizer().fit_transform(target_list)

    # Append a constant feature with value 1 to the end of every input data.
    # Then we do not need to explicitly represent bias - it becomes the last weight.
    data = np.pad(data, [(0, 0), (0, 1)], constant_values=1)

    # Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    # Generate initial model weights.
    weights = generator.uniform(size=[train_data.shape[1], args.classes], low=-0.1, high=0.1)

    for epoch in range(args.epochs):
        permutation = generator.permutation(train_data.shape[0])

        # TODO: Process the data in the order of `permutation`. For every
        # `args.batch_size` of them, average their gradient, and update the weights.
        # You can assume that `args.batch_size` exactly divides `train_data.shape[0]`.
        average_gradient = np.full((weights.shape[0],weights.shape[1]),0)

        for j in range(0, len(train_data), args.batch_size):   
            for i in range(args.batch_size):
                x = train_data[permutation[j + i]]
                sigm = sigmoid(np.dot(x,weights))
                gradient = sigm - train_target[permutation[j + i]]
                gradient = gradient.reshape(1,gradient.shape[0])
                x = x.reshape(x.shape[0],1)
                gradient = np.dot(x,gradient)
                average_gradient = np.add(average_gradient,gradient)
            average_gradient = np.divide(average_gradient,args.batch_size)
            weights = weights - args.learning_rate * average_gradient
            average_gradient = np.full((weights.shape[0],weights.shape[1]),0)



        # TODO: After the SGD epoch, compute the micro-averaged and the
        # macro-averaged F1-score for both the train test and the test set.
        # Compute these scores manually, without using `sklearn.metrics`.
        
        ################# TRAIN SET #######################

        f1_sum = 0
        sum_tp = 0
        sum_fp = 0
        sum_fn = 0
        predictions = []

        for i in range(train_data.shape[0]):
            x = train_data[i]
            estimations = sigmoid(np.dot(x.transpose(),weights))
            pred = values(estimations,train_target[i])
            predictions.append(pred)
        
        for j in range(train_target.shape[1]):
            tp = 0
            fp = 0
            fn = 0
            for i in range(train_target.shape[0]):
                
                if(predictions[i][j] == 1 and train_target[i][j] == 1):
                    tp+=1
                if(predictions[i][j] == 1 and train_target[i][j] == 0):
                    fp+=1
                if(predictions[i][j] == 0 and train_target[i][j] == 1):
                    fn+=1

            denom = tp+fp
            if(denom == 0):
                precision = 0
            else:
                precision = tp/(tp+fp)
            recall = tp/(tp+fn)
            if(precision+recall == 0):
                f1 = 0
            else:
                f1 = (2*precision*recall)/(precision+recall)
            f1_sum += f1
            sum_tp += tp
            sum_fp += fp
            sum_fn += fn
            
        micro_precision = sum_tp/(sum_tp+sum_fp)
        micro_recall = sum_tp/(sum_tp+sum_fn)
        train_f1_micro = (2*micro_precision*micro_recall)/(micro_precision+micro_recall)
        train_f1_macro = f1_sum/train_target.shape[1]
        
        ################## TEST SET ############################

        f1_sum = 0
        sum_tp = 0
        sum_fp = 0
        sum_fn = 0
        predictions = []

        for i in range(test_data.shape[0]):
            x = test_data[i]
            estimations = sigmoid(np.dot(x.transpose(),weights))
            pred = values(estimations,test_target[i])
            predictions.append(pred)
        
        for j in range(test_target.shape[1]):
            tp = 0
            fp = 0
            fn = 0
            for i in range(test_target.shape[0]):
                
                if(predictions[i][j] == 1 and test_target[i][j] == 1):
                    tp+=1
                if(predictions[i][j] == 1 and test_target[i][j] == 0):
                    fp+=1
                if(predictions[i][j] == 0 and test_target[i][j] == 1):
                    fn+=1

            denom = tp+fp
            if(denom == 0):
                precision = 0
            else:
                precision = tp/(tp+fp)
            recall = tp/(tp+fn)
            if(precision+recall == 0):
                f1 = 0
            else:
                f1 = (2*precision*recall)/(precision+recall)
            f1_sum += f1
            sum_tp += tp
            sum_fp += fp
            sum_fn += fn
            
        micro_precision = sum_tp/(sum_tp+sum_fp)
        micro_recall = sum_tp/(sum_tp+sum_fn)
        test_f1_micro = (2*micro_precision*micro_recall)/(micro_precision+micro_recall)
        test_f1_macro = f1_sum/train_target.shape[1]
        

        print("After epoch {}: train F1 micro {:.2f}% macro {:.2f}%, test F1 micro {:.2f}% macro {:.1f}%".format(
            epoch + 1, 100 * train_f1_micro, 100 * train_f1_macro, 100 * test_f1_micro, 100 * test_f1_macro))
    
    return weights, [(100 * train_f1_micro, 100 * train_f1_macro), (100 * test_f1_micro, 100 * test_f1_macro)]


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    weights, metrics = main(args)
    print("Learned weights:",
          *(" ".join([" "] + ["{:.2f}".format(w) for w in row[:10]] + ["..."]) for row in weights.T), sep="\n")
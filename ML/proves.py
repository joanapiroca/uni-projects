#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--dataset", default="wine", type=str, help="Dataset to use")
parser.add_argument("--l2", default=1., type=float, help="L2 regularization factor")
parser.add_argument("--learning_rate", default=0.1, type=float, help="Learning rate")
parser.add_argument("--max_depth", default=None, type=int, help="Maximum decision tree depth")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.25, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
parser.add_argument("--trees", default=1, type=int, help="Number of trees in the forest")
# If you add more arguments, ReCodEx will keep them with your default values.

def softmax(x): #x needs to be a vector and returns a vector
    return np.exp(x)/np.sum(np.exp(x))

#Given one tree and a sample, it returns it's predictions
def predict_class(node, row):
    #print('node',node)
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict_class(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict_class(node['right'], row)
        else:
            return node['right']

#For the 'classes' trees in the t iteration, returns a vector with the prediction for each class of sample row
def predict(row, trees, lr):
    probs = []
    classes = len(trees)
    for k in range(classes):
        prob_k = predict_class(trees[k],row)
        probs.append(lr*prob_k)
    return probs

def build_tree(train, args):
        root = get_split(train, args)
        split(root, int(1), args)
        return root

def test_split(index, value, data): #I have <, is it <=?????
    left, right = [],[]
    left = data[np.array(data)[:,index] < value]
    right = data[np.array(data)[:,index] > value]
    return left, right

#I only mode the g and h for this node, then I need to save indexes not data in the groups
def get_split(data, args):
        b_index, b_value, b_score, b_w, b_groups = 0, 0, -np.inf, [], None
        number_of_features = len(data[0])-3 #-3 because I have targets,g and h in the last columns
        G, H = np.sum(data[:,len(data[0])-3]), np.sum(data[:,len(data[0])-2])

        for index in range(number_of_features):
            #need to sort the data for the k'ths column and apply the same sort to the g's and h's
            sord_indexes_colk = np.argsort(data[:,index])
            datao = data[sord_indexes_colk]
            #g = data[:,len(data[0])-3]
            #h = data[:,len(data[0])-2]
            #G_left, H_left = 0,0 

            ts = np.unique(datao[:, index])

            for j in range(len(ts)-1):    #I put the -1 to not creat empty nodes
                t = (ts[j]+ts[j+1])/2
                groups = test_split(index,t,datao)
                left,right = groups
                G_left = np.sum(left[:,len(left[0])-3])
                H_left = np.sum(right[:,len(right[0])-2])
                G_right =  G - G_left
                H_right = H - H_left

                new_score = (G_left**2/(args.l2 + H_left)) + (G_right**2/(args.l2 + H_right)) - (G**2/(args.l2 + H))                
                if new_score > b_score:
                    w_left = - G_left**2/(args.l2 + H_left)
                    w_right = - G_right**2/(args.l2 + H_right)
                    b_index, b_value, b_score, b_w, b_groups = index, t, new_score, [w_left,w_right], groups
        return {'index':b_index, 'value':b_value,'score':b_score, 'w':b_w, 'groups':b_groups}

def split(node, depth, args):
    max_depth = args.max_depth
    #print('NODE', node)
    left, right = node['groups']
    del(node['groups'])

    if max_depth != None:
        if depth >= max_depth:
            node['left'], node['right'] = (node['w'])[0], (node['w'])[1]
            return

    #if not np.all(left[:,-1] == left[0][-1]):  #crit != 0.0:
    if len(left) > 1: #and not np.all(left[:,-1] == left[0][-1]):
        node['left'] = get_split(left,args)
        split(node['left'], depth+1, args)
    else:
        node['left'] = (node['w'])[0]
    
    #if not np.all(right[:,-1] == right[0][-1]):   #crit != 0.0:
    if len(right) > 1: #and not np.all(right[:,-1] == right[0][-1]):
        node['right'] = get_split(right,args)
        split(node['right'], depth+1, args)
    else:
        node['right'] = (node['w'])[1]



def main(args: argparse.Namespace) -> tuple[list[float], list[float]]:
    # Use the given dataset.
    data, target = getattr(sklearn.datasets, "load_{}".format(args.dataset))(return_X_y=True)

    # Split the data randomly to train and test using `sklearn.model_selection.train_test_split`,
    # with `test_size=args.test_size` and `random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    classes = np.max(target) + 1

    # TODO: Create gradient boosted trees on the classification training data.
    #
    # Notably, train for `args.trees` iteration. During the iteration `t`:
    # - the goal is to train `classes` regression trees, each predicting
    #   a part of logit for the corresponding class.
    # - compute the current predictions `y_{t-1}(x_i)` for every training example `i` as
    #     y_{t-1}(x_i)_c = \sum_{j=1}^{t-1} args.learning_rate * tree_{iter=j,class=c}.predict(x_i)
    #     (note that y_0 is zero)
    # - loss in iteration `t` is
    #     E = (\sum_i NLL(onehot_target_i, softmax(y_{t-1}(x_i) + trees_to_train_in_iter_t.predict(x_i)))) +
    #         1/2 * args.l2 * (sum of all node values in trees_to_train_in_iter_t)
    # - for every class `c`:
    #   - start by computing `g_i` and `h_i` for every training example `i`;
    #     the `g_i` is the first and the `h_i` is the second derivative of
    #     NLL(onehot_target_i_c, softmax(y_{t-1}(x_i))_c) with respect to y_{t-1}(x_i)_c.
    #   - then, create a decision tree minimizing the loss E. According to the slides,
    #     the optimum prediction for a given node T with training examples I_T is
    #       w_T = - (\sum_{i \in I_T} g_i) / (args.l2 + sum_{i \in I_T} h_i)
    #     and the value of the loss with this prediction is
    #       c_GB = - 1/2 (\sum_{i \in I_T} g_i)^2 / (args.l2 + sum_{i \in I_T} h_i)
    #     which you should use as a splitting criterion.
    #
    # During tree construction, we split a node if:
    # - its depth is less than `args.max_depth`
    # - there is more than 1 example corresponding to it (this was covered by
    #     a non-zero criterion value in the previous assignments)

    n = len(train_data)
    #in the last three columns of the data i have g,h and the targets(in this order) to make the splits easier
    g,h = np.zeros((n,1)), np.zeros((n,1))
    train = np.c_[train_data, g, h, train_target]  

    #probability predictions: n x classes matrix with the probabilities for the train_data for eah class (Y in the slides)
    probs_train = np.zeros((len(train_data), classes)) 
    probs_test = np.zeros((len(test_data), classes))

    train_accuracies = []
    test_accuracies = []

    for t in range(args.trees):   
        trees = []  #dimensions 1 x classes where we save all the trees from the iteration t
        #For every class, we train a regression tree 
        for c in range(classes):
            for i in range(n):
                train[i,len(train[0])-3] = ( softmax(probs_train[i])[c] - (train_target[i]==c) )    #g
                train[i,len(train[0])-2] = ( softmax(probs_train[i])[c]  * (1 - softmax(probs_train[i])[c]) )   #h
    
            tree = build_tree(train,args)
            print(tree)
            trees.append(tree)

        #print(trees)

        # - compute the current predictions `y_{t-1}(x_i)` for every training example `i` as y_{t-1}(x_i)_c = \sum_{j=1}^{t-1} args.learning_rate * tree_{iter=j,class=c}.predict(x_i) (note that y_0 is zero)        
        #I only need to predict the probs for the last tree and add it to the last one
        for i in range(len(train_data)):
            probs_train[i] += predict(train_data[i], trees, args.learning_rate)

        pred_train = []
        for i in range(len(train_data)):
            pred_train.append(np.argmax(softmax(probs_train[i])))

        for i in range(len(test_data)):
            probs_test[i] += predict(test_data[i], trees, args.learning_rate)

        pred_test = []
        for i in range(len(test_data)):
            pred_test.append(np.argmax(softmax(probs_test[i])))

        
        train_accuracy = sklearn.metrics.accuracy_score(train_target,pred_train)
        test_accuracy = sklearn.metrics.accuracy_score(test_target,pred_test)

        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

    # TODO: Finally, measure your training and testing accuracies when
    # using 1, 2, ..., `args.trees` of the created trees.
    #
    # To perform a prediction using t trees, compute the y_t(x_i) and return the
    # class with the highest value (pick the smallest class number if there is a tie).


    return [100 * acc for acc in train_accuracies], [100 * acc for acc in test_accuracies]


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    train_accuracies, test_accuracies = main(args)

    for i, (train_accuracy, test_accuracy) in enumerate(zip(train_accuracies, test_accuracies)):
        print("Using {} trees, train accuracy: {:.1f}%, test accuracy: {:.1f}%".format(
            i + 1, train_accuracy, test_accuracy))
#!/usr/bin/env python3
# d9034c78-c8b2-4820-8671-0e919beeaaf3
# 10d1c6c4-6216-4721-b23c-d0de3e0fb138
# c8cb695-9dc2-465b-b4ea-1637eb270929
import argparse

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--bagging", default=False, action="store_true", help="Perform bagging")
parser.add_argument("--dataset", default="wine", type=str, help="Dataset to use")
parser.add_argument("--feature_subsampling", default=1, type=float, help="What fraction of features to subsample")
parser.add_argument("--max_depth", default=None, type=int, help="Maximum decision tree depth")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.25, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
parser.add_argument("--trees", default=1, type=int, help="Number of trees in the forest")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> tuple[float, float]:
    # Use the given dataset.
    data, target = getattr(sklearn.datasets, "load_{}".format(args.dataset))(return_X_y=True)

    # Split the data randomly to train and test using `sklearn.model_selection.train_test_split`,
    # with `test_size=args.test_size` and `random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    # Create random generators.
    generator_feature_subsampling = np.random.RandomState(args.seed)
    def subsample_features(number_of_features: int) -> np.ndarray:
        return generator_feature_subsampling.uniform(size=number_of_features) <= args.feature_subsampling

    generator_bootstrapping = np.random.RandomState(args.seed)
    def bootstrap_dataset(train_data: np.ndarray) -> np.ndarray:
        return generator_bootstrapping.choice(len(train_data), size=len(train_data), replace=True)

    def criterion_value(groups):    
        value = 0.0
        for group in groups:
            if len(group) == 0:
                value.append(0.0)
                continue
            p = np.bincount(group[:, -1].astype(int)).astype(float)
            p /= len(group)
            p = p[p != 0]
            value -= len(group) * np.sum(p * np.log(p))
        return value

    def test_split(index, value, data): 
        left, right = [],[]
        left = data[data[:,index] < value]
        right = data[data[:,index] > value]
        return left, right

    def get_split(data):
        b_index, b_value, b_score, b_groups = 0, 0, float('+inf'), None
        number_of_features = len(data[0])-1 #-1 because I have targets in the last column
        sub_feat = subsample_features(number_of_features)

        for index in range(len(data[0])-1): 
            if sub_feat[index]:
                ts = np.unique(data[:, index])
                for i in range(len(ts)-1):
                    t = (ts[i] + ts[i + 1])/2
                    groups = test_split(index, t, data)
                    crit_value = criterion_value(groups)
                    #print(crit_value)
                    if  crit_value < b_score:
                        b_index, b_value, b_score, b_groups = index, t, crit_value, groups
        return {'index':b_index, 'value':b_value,'score':b_score, 'groups':b_groups}

    def to_terminal(group):
        targets = group[:,-1].astype(int) 
        return np.argmax(np.bincount(targets))
    
    def split(node, depth, args):
        max_depth = args.max_depth
        #print('iteration split in depth',depth)
        left, right = node['groups']
        del(node['groups'])

        if len(left)==0 or len(right)==0: 
            node = to_terminal(np.vstack((left,right)))
            return

        if max_depth != None:
            if depth >= max_depth:
                node['left'], node['right'] = to_terminal(left), to_terminal(right)
                return

        #groups = [],left
        #crit = criterion_value(groups)
        if not np.all(left[:,-1] == left[0][-1]):  #crit != 0.0:
        #if (node['score'])[0] != 0:
            node['left'] = get_split(left)
            split(node['left'], depth+1, args)
        else:
            node['left'] = to_terminal(left)
        
        #groups = [],right
        #crit = criterion_value(groups)
        if not np.all(right[:,-1] == right[0][-1]):   #crit != 0.0:
        #if (node['score'])[1] != 0:
            node['right'] = get_split(right)
            split(node['right'], depth+1, args)
        else:
            node['right'] = to_terminal(right)
    
    def predict(node, row):
        if row[node['index']] < node['value']:
            if isinstance(node['left'], dict):
                return predict(node['left'], row)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return predict(node['right'], row)
            else:
                return node['right']
    
    def build_tree(train, args):
        groups = train,[]
        root = get_split(train)
        split(root, int(1), args)
        #print(root)
        return root


    # TODO: Create a random forest on the training data.
    #
    # Use a simplified decision tree from the `decision_tree` assignment:
    # - use `entropy` as the criterion
    # - use `max_depth` constraint, to split a node only if:
    #   - its depth is less than `args.max_depth`
    #   - the criterion is not 0 (the corresponding instance targets are not the same)  
    # When splitting nodes, proceed in the depth-first order, splitting all nodes
    # in the left subtree before the nodes in right subtree.
    #
    # Additionally, implement:
    # - feature subsampling: when searching for the best split, try only
    #   a subset of features. Notably, when splitting a node (i.e., when the
    #   splitting conditions [depth, criterion != 0] are satisfied), start by
    #   generating a feature mask using
    #     subsample_features(number_of_features)
    #   which gives a boolean value for every feature, with `True` meaning the
    #   feature is used during best split search, and `False` it is not
    #   (i.e., when `feature_subsampling == 1`, all features are used).
    #
    # - train a random forest consisting of `args.trees` decision trees
    #
    # - if `args.bagging` is set, before training each decision tree
    #   create a bootstrap sample of the training data by calling
    #     dataset_indices = bootstrap_dataset(train_data)
    #   and if `args.bagging` is not set, use the original training data.
    #
    # During prediction, use voting to find the most frequent class for a given
    # input, choosing the one with the smallest class number in case of a tie.

    train = np.c_[train_data,train_target]  
    k = 0 #number of trees in the forest
    #pred_forest_train = [] #Each row contains predictions trained one each tree of the forest
    #pred_forest_test = []
    trees = []

    while k < args.trees:
        #print('new tree')
        if  args.bagging == True:
            dataset_indices = bootstrap_dataset(train_data)
            tree = build_tree(train[dataset_indices],args)
        else: 
            tree = build_tree(train,args)
        #print('obtained tree',tree)
        trees.append(tree)
    
        k += 1

    #PREDICTION VOTING
    pred_train = []
    for row in train_data:
        pred_row = []
        for tree in trees:
            pred_row.append(predict(tree,row))
        pred_train.append(np.argmax(np.bincount(pred_row)))
    
    pred_test = []
    for row in test_data:
        pred_row = []
        for tree in trees:
            pred_row.append(predict(tree,row))
        pred_test.append(np.argmax(np.bincount(pred_row)))

    
    # TODO: Finally, measure the training and testing accuracy.
    
    train_accuracy = sklearn.metrics.accuracy_score(train_target,pred_train)
    test_accuracy = sklearn.metrics.accuracy_score(test_target,pred_test)
    '''
    count = 0
    for i in range(len(train_data)):
        if train_target[i] == pred_train[i]: count += 1
    train_accuracy = count/len(train_data)

    count = 0
    for i in range(len(test_data)):
        if test_target[i] == pred_test[i]: count += 1
    test_accuracy = count/len(test_data)
    '''
    return 100 * train_accuracy, 100 * test_accuracy


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    train_accuracy, test_accuracy = main(args)

    print("Train accuracy: {:.1f}%".format(train_accuracy))
    print("Test accuracy: {:.1f}%".format(test_accuracy))
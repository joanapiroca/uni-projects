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
parser.add_argument("--criterion", default="gini", type=str, help="Criterion to use; either `gini` or `entropy`")
parser.add_argument("--dataset", default="wine", type=str, help="Dataset to use")
parser.add_argument("--max_depth", default=None, type=int, help="Maximum decision tree depth")
parser.add_argument("--max_leaves", default=None, type=int, help="Maximum number of leaf nodes")
parser.add_argument("--min_to_split", default=2, type=int, help="Minimum examples required to split")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.25, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
# If you add more arguments, ReCodEx will keep them with your default values.

def criterion_value(groups,args):    #TABULATION????
    # count all samples at split point
    # sum weighted Gini index for each group
    criterion_type = args.criterion
    value = 0.0
    for group in groups:
        # score the group based on the score for each class
        p = np.bincount(group[:, -1].astype(int)).astype(float)
        p /= np.sum(p)
        p = p[p != 0]
        if criterion_type == "gini":
            value += len(group) * np.sum(p * (1 - p))
        else:
            value -= len(group) * np.sum(p * np.log(p))

    if args.max_leaves != None:
        node = np.vstack((groups))
        p = np.bincount(node[:, -1].astype(int)).astype(float)
        p /= np.sum(p)
        p = p[p != 0]
        if criterion_type == "gini":
            value -= len(node) * np.sum(p * (1 - p))
        else:
            value += len(node) * np.sum(p * np.log(p))    
    return value

def test_split(index, value, data): #I have <, is it <=?????
    left, right = [],[]
    left = data[np.array(data)[:,index] < value]
    right = data[np.array(data)[:,index] > value]
    return left, right

def get_split(data,args):
    b_index, b_value, b_score, b_groups = float('+inf'), float('+inf'), float('+inf'), None
    for index in range(len(data[0])-2): #-2 because i have the target in the last column and I don't wanna go to the last feature of the data inputs
        ts = np.unique(data[:, index])
        for i in range(len(ts)-1):
            t = (ts[i] + ts[i + 1])/2
            groups = test_split(index, t, data)
            crit_value = criterion_value(groups,args)
            if crit_value < b_score:
                b_index, b_value, b_score, b_groups = index, t, crit_value, groups
    return {'index':b_index, 'value':b_value,'score':b_score, 'groups':b_groups}

# Create a terminal node value
def to_terminal(group):
    targets = np.array(group)[:,-1].astype(int) #[row[-1] for row in group]
    return np.argmax(np.bincount(targets))


# Create child splits for a node or make terminal
def split(node, depth, args, leaves,root,scores):
    max_leaves, max_depth, min_size = args.max_leaves,args.max_depth,args.min_to_split
    left, right = node['groups']
    del(node['groups'])
    # check for a no split: ?????
    if len(left)==0 or len(right)==0:   
        node['left'] = node['right'] = to_terminal(np.vstack((left,right)))
        #leaves += 2
        return
    
    #check for max leaves
    if max_leaves == None:
        #del(node['groups'])
        # check for max depth
        if max_depth != None:
            if depth >= max_depth:
                node['left'], node['right'] = to_terminal(left), to_terminal(right)
                return
        # process left child (DEPTH-FIRST TO THE LEFT)
        if len(left) <= min_size:
            node['left'] = to_terminal(left)
        else:
            node['left'] = get_split(left,args)
            split(node['left'], depth+1, args,0,root,scores)
        # process right child (DEPTH-FIRST TO THE RIGHT)
        if len(right) <= min_size:
            node['right'] = to_terminal(right)
        else:
            node['right'] = get_split(right,args)
            split(node['right'], depth+1, args,0,root,scores)

    else: # MAX LEAVES: split the leaf that minimizes ctl+tcr-ct (THINK HOW TO REMEMBER THE REMAINING LEAVES)
        #del(node['groups'])
        if depth != max_depth:
            if len(left) < min_size:
                node['left'] = to_terminal(left)
        #        print('MIN SIZE LEFT')
                #leaves+=1 I've already counted this leave when I called the split function and created the leaf
            else: 
                node['left'] = get_split(left,args)
                scores.append(node['left']['score'])

            if len(right) < min_size:
                node['right'] = to_terminal(right)
        #        print('MIN SIZE RIGHT')
            else:
                node['right'] = get_split(right,args)
                scores.append(node['right']['score']) 
        else: #reached max_depth
        #    print('MAX DEPTH REACHED')
            node['left'], node['right'] = to_terminal(left), to_terminal(right)

        if leaves == max_leaves:
            tree_to_terminal(root)
            return   #and finish!!! how do I do it???   MIER           

        #print(scores)
        new_score = np.min(scores)
        #print(new_score)
        #print('leaves number',leaves)
        scores.pop(np.argmin(scores))
        #print('root',root)
        leave_to_split = best_split(root,new_score)
        if leave_to_split == None: 
            tree_to_terminal(root)
        #    print('none')
        else: split(leave_to_split, depth+1,args,leaves+1,root,scores)

def best_split(node, best_score):  #For finding the best leave to split in max_leaves case
    #print('best split',node)
    if type(node['left']) != np.int64:        
        if 'groups' not in node['left']:
    #        print('next node LEFT')
            if best_split(node['left'], best_score) != None:    #If I don't put this line, the function neves goes to the right branch of the tree
                return best_split(node['left'], best_score)
        else:
    #        print('leave')
            if node['left']['score'] == best_score:
    #            print('best score found')
                return node['left']
    if type(node['right']) != np.int64:
    #    print('RIIIGHT')
        if 'groups'  not in node['right']:
    #        print('next node RIGHT')
            return best_split(node['right'], best_score)
        else:
    #        print('leave')
            if node['right']['score'] == best_score:
    #            print('best score found')
                return node['right']
    
def tree_to_terminal(root): #CANVIAR A HIPOTETICO
    #left, right = root['groups']
    #print('tree to terminal',root)
    if type(root['left']) != np.int64:
        if 'groups' not in root['left']:
            tree_to_terminal(root['left'])
        else:
            left, right = root['left']['groups']
            del((root['left'])['groups'])
            root['left'] = to_terminal(np.vstack((left,right)))
    
    if type(root['right']) != np.int64:
        if 'groups' not in root['right']:
            tree_to_terminal(root['right'])
        else:
            left, right = root['right']['groups']
            del((root['right'])['groups'])
            root['right'] = to_terminal(np.vstack((left,right)))   

# Make a prediction with a decision tree
def predict(node, row):
    #print('predict',node)
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
        root = get_split(train,args)
        scores = []        
        split(root, int(1), args,int(2),root,scores)
        return root

def main(args: argparse.Namespace) -> tuple[float, float]:
    # Use the given dataset.
    data, target = getattr(sklearn.datasets, "load_{}".format(args.dataset))(return_X_y=True)

    # Split the data randomly to train and test using `sklearn.model_selection.train_test_split`,
    # with `test_size=args.test_size` and `random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)    

    # TODO: Manually create a decision tree on the training data.
    #
    # - For each node, predict the most frequent class (and the one with
    #   smallest number if there are several such classes).
    #
    # - When splitting a node, consider the features in sequential order, then
    #   for each feature consider all possible split points ordered in ascending
    #   value, and perform the first encountered split decreasing the criterion
    #   the most. Each split point is an average of two nearest unique feature values
    #   of the instances corresponding to the given node (e.g., for four instances
    #   with values 1, 7, 3, 3 the split points are 2 and 5).
    #
    # - Allow splitting a node only if:
    #   - when `args.max_depth` is not `None`, its depth must be less than `args.max_depth`;
    #     depth of the root node is zero;
    #   - there are at least `args.min_to_split` corresponding instances;
    #   - the criterion value is not zero.
    #
    # - When `args.max_leaves` is `None`, use recursive (left descendants first, then
    #   right descendants) approach, splitting every node if the constraints are valid.
    #   Otherwise (when `args.max_leaves` is not `None`), always split a node where the
    #   constraints are valid and the overall criterion value ($c_left + c_right - c_node$)
    #   decreases the most. If there are several such nodes, choose the one
    #   which was created sooner (a left child is considered to be created
    #   before a right child).

    #MAX_LEAVES MISSING   , TARGETS MISSING, CRITERIA ENTROPY MISSING, 
    train = np.c_[train_data,train_target]  #train data matrix and train target in the last column to construct the tree
    tree = build_tree(train, args)
    #print(tree)

    # TODO: Finally, measure the training and testing accuracy.
    #Predictions
    pred_train = []
    for row in train_data:
        prediction = predict(tree, row)
        pred_train.append(prediction)

    pred_test = []
    for row in test_data:
        prediction = predict(tree, row)
        pred_test.append(prediction)

    count = 0
    for i in range(len(train_data)):
        if train_target[i] == pred_train[i]: count += 1
    train_accuracy = count/len(train_data)

    count = 0
    for i in range(len(test_data)):
        if test_target[i] == pred_test[i]: count += 1
    test_accuracy = count/len(test_data)   

    return 100 * train_accuracy, 100 * test_accuracy


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    train_accuracy, test_accuracy = main(args)

    print("Train accuracy: {:.1f}%".format(train_accuracy))
    print("Test accuracy: {:.1f}%".format(test_accuracy))
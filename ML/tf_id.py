#!/usr/bin/env python3
# d9034c78-c8b2-4820-8671-0e919beeaaf3
# 10d1c6c4-6216-4721-b23c-d0de3e0fb138
# c8cb695-9dc2-465b-b4ea-1637eb270929
import argparse
import lzma
import pickle
import os
import sys
import urllib.request

import numpy as np
import sklearn.metrics
import sklearn.model_selection
import sklearn.neighbors
import re

from collections import defaultdict
from collections import Counter

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--idf", default=False, action="store_true", help="Use IDF weights")
parser.add_argument("--k", default=1, type=int, help="K nearest neighbors to consider")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=177, type=int, help="Random seed")
parser.add_argument("--tf", default=False, action="store_true", help="Use TF weights")
parser.add_argument("--test_size", default=1000, type=int, help="Test set size")
parser.add_argument("--train_size", default=1000, type=int, help="Train set size")
# For these and any other arguments you add, ReCodEx will keep your default value.


class NewsGroups:
    def __init__(self,
                 name="20newsgroups.train.pickle",
                 data_size=None,
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2223/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename=name)

        with lzma.open(name, "rb") as dataset_file:
            dataset = pickle.load(dataset_file)

        self.DESCR = dataset.DESCR
        self.data = dataset.data[:data_size]
        self.target = dataset.target[:data_size]
        self.target_names = dataset.target_names

def main(args: argparse.Namespace) -> float:
    # Load the 20newsgroups data.
    newsgroups = NewsGroups(data_size=args.train_size + args.test_size)

    # Create train-test split.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        newsgroups.data, newsgroups.target, test_size=args.test_size, random_state=args.seed)

    # TODO: Create a feature for every term that is present at least twice
    # in the training data. A term is every maximal sequence of at least 1 word character,
    # where a word character corresponds to a regular expression `\w`.
    print('original data:',train_data) #
    shape_train = len(train_data)
    train_data_ = [re.findall('\w+', i) for i in train_data]
    print('find words data:',train_data) #
    words = []
    for i in train_data_:
        for j in i:
            words.append(j)
    counter = Counter(words)
    w = []
    for elem in list(counter.items()):
        if(elem[1] > 1):
            w.append(elem[0])


    # TODO: For each document, compute its features as
    # - term frequency(TF), if `args.tf` is set;
    # - otherwise, use binary indicators (1 if a given term is present, else 0)
    #
    # Then, if `args.idf` is set, multiply the document features by the
    # inverse document frequencies (IDF), where
    # - use the variant which contains `+1` in the denominator;
    # - the IDFs are computed on the train set and then reused without
    #   modification on the test set.

    features = np.zeros((shape_train,len(w)))
    term_document_counters = np.zeros(len(w))
    features_test = np.zeros((len(test_data),len(w)))
    test_data_ = [re.findall('\w+', i)for i in test_data]
    idf = []

    ############## Training ######################

    for i in range(shape_train):
            document = train_data_[i]
            d_len = len(document)
            words_counter = Counter(document)
            if(i < len(test_data_)):
                document_test = test_data_[i]
                d_len_test = len(document_test)
                words_counter_test = Counter(document_test)
            for j in range(len(w)):
                word = w[j]
                t = words_counter[word]
                if(t>0): term_document_counters[j] += 1
                
                if(i < len(test_data_)):
                    t_test = words_counter_test[word]
                    if args.tf:
                        features_test[i][j] = (t_test/d_len_test)
                    else:
                        if(t_test>0):
                            features_test[i][j] = 1
                if args.tf:
                    features[i][j] = (t/d_len)
                else:
                    if(t>0):
                        features[i][j] = 1

    if args.idf:   
        for d in term_document_counters:
            idf.append(np.log(shape_train/(d+1)))
        
        for i in range(shape_train):
            features[i] = np.multiply(features[i],idf)
            if(i < len(test_data)):
                features_test[i] = np.multiply(features_test[i],idf)

    

    

    # TODO: Perform classification of the test set using the k-NN algorithm
    # from sklearn (pass the `algorithm="brute"` option), with `args.k` nearest
    # neighbors. For TF-IDF vectors, the cosine similarity is usually used, where
    #   cosine_similarity(x, y) = x^T y / (||x|| * ||y||).
    #
    # To employ this metric, you have several options:
    # - you could try finding out whether `KNeighborsClassifier` supports it directly;
    # - or you could compute it yourself, but if you do, you have to precompute it
    #   in a vectorized way, so using `metric="precomputed"` is fine, but passing
    #   a callable as the `metric` argument is not (it is too slow);
    # - finally, the nearest neighbors according to cosine_similarity are equivalent to
    #   the neighbors obtained by the usual Euclidean distance on L2-normalized vectors.

    kn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=args.k,algorithm="brute",metric="cosine")
    kn.fit(features,train_target)
    

    '''
    for i in range(len(test_data_)):
            document = test_data_[i]
            words_counter = Counter(document)
            for j in range(len(w)):
                word = w[j]
                t = words_counter[word]
                if args.tf:
                    features_training[i][j] = (t/d_len)
                else:
                    if(t>0):
                        features_training[i][j] = 1
    
    if args.idf:
        for i in range(len(test_data)):
            features_training[i] = np.multiply(features_training[i],idf)
    '''
    predicts = kn.predict(features_test)


    # TODO: Evaluate the performance using a macro-averaged F1 score.
    
    f1_score =  sklearn.metrics.f1_score(test_target,predicts,average='macro')

    return f1_score


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    f1_score = main(args)
    print("F-1 score for TF={}, IDF={}, k={}: {:.1f}%".format(args.tf, args.idf, args.k, 100 * f1_score))
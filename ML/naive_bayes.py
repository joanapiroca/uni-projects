#!/usr/bin/env python3
# d9034c78-c8b2-4820-8671-0e919beeaaf3
# 10d1c6c4-6216-4721-b23c-d0de3e0fb138
# c8cb695-9dc2-465b-b4ea-1637eb270929
import argparse

import numpy as np
import scipy.stats

import sklearn.datasets
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--alpha", default=0.1, type=float, help="Smoothing parameter of our NB classifier")
parser.add_argument("--naive_bayes_type", default="gaussian", type=str, help="NB type gaussian/multinomial/bernoulli")
parser.add_argument("--classes", default=10, type=int, help="Number of classes")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=72, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> float:
    # Load the digits dataset.
    data, target = sklearn.datasets.load_digits(n_class=args.classes, return_X_y=True)

    # Split the dataset into a train set and a test set.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    # TODO: Train a naive Bayes classifier on the train data.
    #
    # The `args.naive_bayes_type` can be one of:
    # - "gaussian": implement Gaussian NB training, by estimating mean and
    #   variance of the input features. For variance estimation use
    #     1/N * \sum_x (x - mean)^2
    #   and additionally increase all estimated variances by `args.alpha`.
    #
    #   During prediction, you can compute the probability density function
    #   of a Gaussian distribution using `scipy.stats.norm`, which offers
    #   `pdf` and `logpdf` methods, among others.
    #
    # - "multinomial": Implement multinomial NB with smoothing factor `args.alpha`.
    #
    # - "bernoulli": Implement Bernoulli NB with smoothing factor `args.alpha`.
    #   Because Bernoulli NB works with binary data, binarize the features as
    #   [feature_value >= 8], i.e., consider a feature as one iff it is >= 8,
    #   during both estimation and prediction.
    #
    # In all cases, the class prior is the distribution of the train data classes.
    K = args.classes #number of classes from 0 to K-1
    (N,D) = np.shape(train_data) #number of samples and number of features
    
    Nks = [] #to know how many samples of each class in prther to compute the prior distribution of Ck
    mu,var = [],[] #K x D matrix where each element i,j is the mu (or var) for class i and feature (pixel) j.
    if args.naive_bayes_type == "gaussian":
        for k in range(K):
            x_k = [] #samples from train_data wich have target k
            for i in range(N): 
                if train_target[i] == k:
                    x_k.append(train_data[i])
            Nk = len(x_k)
            Nks.append(Nk)
            mu_k = (1/Nk)*np.sum(x_k,axis=0) #lenght D, for all k's
            mu.append(mu_k)
            dif_squared = [np.power(x_k[j]-mu_k,2) for j in range(Nk)]
            var.append( (1/Nk)*np.sum(dif_squared,axis=0)+args.alpha ) 


    p = [] #K x D matrix
    if args.naive_bayes_type == "bernoulli":
        for k in range(K): #number of documents of class k with non-zero feature d
            pos = []
            for i in range(N):
                if train_target[i] == k: pos.append([(train_data[i][j] >= 8) for j in range(D)])
            Nk = len(pos)
            Nks.append(Nk)
            p.append( (1/(Nk+2*args.alpha))*(np.sum(pos,axis=0)+args.alpha) )          

    if args.naive_bayes_type == "multinomial":
        p = [] #K x D matrix with the estimators
        for k in range(K):
            x_k = [] #samples from train data that are target class k
            for i in range(N):
                if train_target[i] == k: x_k.append(train_data[i])
            n_dk = np.sum(x_k,axis=0)
            Nks.append(len(x_k))
            p.append( (n_dk+args.alpha)/(np.sum(x_k)+args.alpha*D) )  #sum(i)sum(d')xid' is the sum of all the elements in matrix x_k


    # TODO: Predict the test data classes and compute the test accuracy.
    #PREDICTIONS: WHAT IS THE PROBABILITY OF CK???? I'M ASSUMING IT'S Nk/N????? but from the train data!!!
    #To know the probability of each class, i need to know hoy many elements are in each class:
    '''
    #Nks FROM TRAIN DATA NOT FROM HERE FOR THE PRIOR DISTRIBUTION
    Nks = np.zeros(K)
    for t in test_target:
        Nks[t] += 1
    '''
    N_test = len(test_data)
    if args.naive_bayes_type == "bernoulli":
        pred = np.zeros(len(test_data))
        for i in range(len(test_data)):
            probs = []
            for k in range(K):
                xw = np.sum([ ( (test_data[i][d]>=8)*np.log(p[k][d]/(1-p[k][d]))+np.log(1-p[k][d]) ) for d in range(D)])
                probs.append( (np.log(Nks[k]/N)+xw) )
            pred[i] = np.argmax(probs)

    if args.naive_bayes_type == "multinomial":
        pred = np.zeros(len(test_data))
        for i in range(len(test_data)):
            probs = []
            for k in range(K):
                xw = np.sum([ ( test_data[i][d]*np.log(p[k][d]) ) for d in range(D)])
                probs.append( (np.log(Nks[k]/len(train_data))+xw)  )
            pred[i] = np.argmax(probs)

    if args.naive_bayes_type == "gaussian":
        pred = np.zeros(len(test_data))
        for i in range(len(test_data)):
            probs = []
            for k in range(K):
                log_prob_xi_ifCk = np.sum(scipy.stats.norm.logpdf(test_data[i],loc=mu[k],scale=np.sqrt(var[k])))
                probs.append(log_prob_xi_ifCk + np.log(Nks[k]/N))
            pred[i] = np.argmax(probs)      
    
    count = 0
    for i in range(len(test_data)):
        if test_target[i] == pred[i]: count += 1
    test_accuracy = count/len(test_data)

    return test_accuracy


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    test_accuracy = main(args)

    print("Test accuracy {:.2f}%".format(100 * test_accuracy))
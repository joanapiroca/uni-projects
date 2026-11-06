import numpy as np
import matplotlib.pyplot as plt

class Perceptron:

    def predict(self, X):
        # Compute output of the perceptron.
        assert self.weights is not None

        # If input X is one sample, we append the treshold value and return 1/0 
        if(len(X.shape) == 1):
            X_extended = np.append(X, [1])
            return 1 if sum(X_extended * self.weights) > 0 else 0

        # Returns one value 0/1 if the input was a single vector or a vector with values 0/1 with the output of the entire perceptron for all samples in X
        X_extended = np.concatenate((X, np.ones((X.shape[0],1))), axis=1)
        potential = self.weights @ X_extended.T
        result = np.where(potential > 0, 1, 0)
        return result  
            
    def fit(self, X, y, lr=None, max_epochs=None):
        # trains perceptron using perceptron learning algorithm
        # on the training set `X` (two-dimensional numpy array of floats) with 
        # the desired outputs `y` (vector of integers 0/1). If the learning rate `lr == None`, 
        # `self.lr` is used. If `max_epochs == None`, self.max_epochs is used. 
        # Returns the number of epochs used in the training (at most `max_epochs`).
        epochs = max_epochs if max_epochs != None else self.max_epochs
        lr = lr if lr != None else self.lr
        for i in range(epochs):
            if(self.score(X,y) == 0.0):
                return i
            self.partial_fit(X,y,lr)
        return epochs
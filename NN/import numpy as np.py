import numpy as np
import sklearn.neural_network

def sigmoid(z):
    return 1.0/(1 + np.exp(-z))


x = [1,0]

for i in range(2):
    x[0] = i
    for j in range(2):
        x[1] = j

        weights_x_h1 = [0.98408437, 0.98408437]
        weights_x_h2 = [0.97979438, 1.02020562]

        weights_h_o1 = [0.9363375,0.9363375 ]
        weights_h_o2 = [-0.91917754,1.08082246]

        b_h1 = 0
        b_h2 = 0
        b_o1 = 0.95
        b_o2 = 0.75

        h1 = weights_x_h1[0]*x[0] + weights_x_h1[1]*x[1] + b_h1
        h2 = weights_x_h2[0]*x[0] + weights_x_h2[1]*x[1] + b_h2

        h1 = sigmoid(h1)
        h2 = sigmoid(h2)

        o1 = weights_h_o1[0]*h1 + weights_h_o1[1]*h2 + b_o1
        o2 = weights_h_o2[0]*h1 + weights_h_o2[1]*h2 + b_o2

       
        print("H1: ",h1," H2: ",h2, " O1: ",o1, "O2: ", o2)
        o1 = sigmoid(o1)
        o2 = sigmoid(o2)

        

        if(o1 >= 0.5):
            o1 = 1
        else: o1 = 0

        print('valor del sigmoide de o2:',o2)
        if(o2 >= 0.5):
            o2 = 1
        else: o2 = 0

        print("INPUTS: :", x," OUTPUTS: ",o1,o2," EXPECTED: ", i+j+1)






import numpy as np

def sigmoid(z,slope):
    return 1.0/(1 + np.exp(-slope*z))

w_i_hb = np.array([[1,1],[1,1],[0,0]])
w_h_ob = np.array([[1,1],[-1, 1],[-0.55,-0.75]])
p = np.array([[0,0]])
d = np.array([[0, 1]])
lr = 1.5
slope = 1

### Compute output for p ### 
y_h = sigmoid(np.append(p, np.ones((p.shape[0],1)),axis=1)@w_i_hb, slope)
y_o = sigmoid(np.append(y_h,np.ones((p.shape[0],1)),axis=1)@w_h_ob,slope)

#Backpropagation updating weights:
delta_o = (d[0]-y_o[0])*slope*y_o[0]*(1-y_o[0])
w_h_ob1 = w_h_ob + (lr*np.append(delta_o*y_h,1)).reshape(-1,1)

delta_h = np.append(delta_o,1)*w_h_ob.T*slope*np.append(y_h,1)*(1-np.append(y_h,1))
w_i_hb1 = w_i_hb + lr * (delta_h*np.append(y_h,1)).transpose()

print(" The updated weight matrix of weights between the input and the hidden layer is: ", w_i_hb1[:2])
print("The updated biases for the hidden layer are: ", w_i_hb1[2])
print(" The updated weight matrix of weights between the hidden and the output layer is: ", w_h_ob1[:2])
print("The updated biases for the output layer are: ", w_h_ob1[2])


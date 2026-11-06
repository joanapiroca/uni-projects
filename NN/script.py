import numpy as np
import matplotlib.pyplot as plt


import sklearn.neural_network
import sklearn.metrics

npz_file = np.load('FMNISTSmall.npz')
train_img = npz_file['trainImgSmall']
train_lab = npz_file['trainLabelSmall']
test_img = npz_file['testImgSmall']
test_lab = npz_file['testLabelSmall']

def task1():

    regressor = sklearn.neural_network.MLPRegressor(20)
    regressor.fit(train_img,train_lab)
    predictions = regressor.predict(test_img)
    for i in range(len(predictions)):
        if(predictions[i] < 0):
            predictions[i] = 0
        elif(predictions[i] > 9):
            predictions[i] = 9
        else:
            predictions[i] = round(predictions[i],0)

    print(sklearn.metrics.accuracy_score(test_lab,predictions)*100)

def task2():
    
    classifier = sklearn.neural_network.MLPClassifier((95,100))
    classifier.fit(train_img, train_lab)
    predictions = classifier.predict(test_img)
    print(classifier.n_layers_)
    print(sklearn.metrics.accuracy_score(test_lab,predictions)*100)

    neurons = []
    for i in range(1,101):
        for j in range(1,101):
            neurons.append((i,j))

    parameters = {'hidden_layer_sizes': neurons}
    mlp_class = sklearn.neural_network.MLPClassifier()
    grid_search = sklearn.model_selection.GridSearchCV(mlp_class,parameters)
    grid_search.fit(train_img,train_lab)
    print(grid_search)

    import pickle
    # now you can save it to a file
    with open('filename.pkl', 'wb') as f:
        pickle.dump(grid_search, f)


    '''
    images = []
    predicted_value = []
    correct_value = []
    cont = 0
    for i in range(len(predictions)):
        if(predictions[i] != test_lab[i]):
            cont+=1
            images.append(test_img[i])
            predicted_value.append(predictions[i])
            correct_value.append(test_lab[i])
        if(cont == 9): 
            break

    for i in range(len(images)):
        plt.figure()
        plt.imshow(images[i,:].reshape(28,28), cmap='gray')
        print("Predicted: " + predicted_value[i] + " Expected: " + correct_value[i])
    '''


        
task1()
task2()

'''

Task 3: The approach used in task 2 is better than task 1 because we are performing
a multiclass classification task, not a regression one. We are not predicting a single value
given an image, we are trying to classify that image in one of 9 different pre-defined classes. If we use
MLPRegressor we are oblied to round the predicted values to get the class, as we do that, we are forcing
missclassification mistakes.

'''


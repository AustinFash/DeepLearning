import math
import numpy as np  
from download_mnist import load
import operator  
import time
import os

# Loading MNIST dataset 
x_train, y_train, x_test, y_test = load()

# Reshape the images into a 28x28 and making them float type 
x_train = x_train.reshape(60000, 28, 28).astype(float)
x_test  = x_test.reshape(10000, 28, 28).astype(float)

# Set value of K
k = 10

def kNNClassify(newInput, dataSet, labels, k):
    result = []
    ########################
    # Input your code here #
    ########################
    
    # For each test image in newInput:
    for test_img in newInput:

        # Compute the L1 (Manhattan) distance
        distances = np.sum(np.abs(dataSet - test_img), axis=(1,2))
        
        # Compute the L2 (Euclidean) distance 
        distances = np.sqrt(np.sum((dataSet - test_img) ** 2, axis=(1, 2)))

        # Identify the indices of the k nearest neighbors.
        k_indices = np.argsort(distances)[:k]

        # Retrieve the corresponding labels.
        k_labels = labels[k_indices]

        # Use majority voting to decide the label.
        predicted_label = np.bincount(k_labels).argmax()
        result.append(predicted_label)

    ####################
    # End of your code #
    ####################
    return np.array(result)

start_time = time.time()
outputlabels = kNNClassify(x_test[0:1000], x_train, y_train, k)

# Compute the accuracy on the first 1000 test samples.
result = y_test[0:1000] - outputlabels
accuracy = (1 - np.count_nonzero(result) / len(outputlabels))


print ("---classification accuracy for knn on mnist: %s ---" % accuracy)
print ("---execution time: %s seconds ---" % (time.time() - start_time))

# K-Nearest-Neighbors
MATLAB function to perform KNN on a set of training/testing data

The code can be run from the command prompt with the following structure:

[y_hat, labels] = KNN('MNIST.mat',7, 'numTestInstances', 100, ...
    'varianceWeighting',true);

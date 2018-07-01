function [y_hat, labels] = KNN(dataset,k,varargin)
% Description: Performs k-nearest neighbors on the dataset identified by
% 'filename'
%
% INPUTS
% dataset: .mat file where data is stored[string literal]
% k: KNN hyperparameter
%
% OUTPUTS:
% y_hat: vector containing estimated output classes [n_test_data x 1]
% labels: matrix containing one-hot encoded y_hat [n_test_data x n_classes]

% Written by: Joshua Gafford
% Date: 06/21/2018

% format for dataset .mat file:
% dataset: structure containing input data. The struct must have the
% following elements:
%
%           data.input_count: a [1x1] scalar containing number of input 
%               features (m features)
%           data.output_count: a [1x1] scalar containging number of output
%               classes (k classes)
%
%           data.training_count: a [1x1] scalar containing number of data
%               points in training set (n_train)
%           data.test_count:  a [1x1] scalar containing number of data 
%               points in test set (n_test)
%           data.validation_count: a [1x1] scalar containing number of data
%               points in validation set (n_val)
%
%           data.training.input: a [n_train x m] array containing inputs of
%               the training set (m features, n_train data points)
%           data.training.output: a [n_train x k] array containing outputs 
%               of the training set (one-hot vectorized, n_train data 
%               points, k features)
%           data.training.classes: a [n_train x 1 ] array containing output
%               classes of the training set (n_train data points)
%
%           data.test.input: a [n_test x m] array containing inputs of the 
%               test set (m features, n_test data points)
%           data.test.output: a [n_test x k] array containing outputs of 
%               the test set (one-hot vectorized, n_test data points, 
%               k features)
%           data.test.classes: a [n_test x 1] array containing output 
%               classes of the test set (non-vectorized, n_test data points)
%
%           data.validation.input: a [n_val x m] array containing inputs of
%               the validation set (m features, n_val data points)
%           data.validation.output: a [n_val x k] array containing outputs
%               of the validation set (one-hot vectorized, n_val data 
%               points, k features)
%           data.validation.classes:  a [n_val x 1] array containing output
%               classes of the validation set (non-vectorized, n_val data
%               points)

% Parse inputs

p = inputParser;
addRequired(p, 'dataset', @ischar);
addOptional(p, 'numTestInstances', 1, ...
    @(x) isnumeric(x) && isscalar(x) && (x>0));
addOptional(p, 'varianceWeighting', false,...
    @(x) islogical(x));

parse(p,dataset,varargin{:});

n_test_instances = p.Results.numTestInstances;
weighting = p.Results.varianceWeighting;


data = load(dataset);


% Load training data
x_train = data.training.input;
y_train = data.training.output;
n_train_data = length(x_train);

% Load testing data
x_test = data.test.input;
% y_test = data.test.output;
y_classes = data.test.classes;

if n_test_instances>length(y_classes)
    msg = 'Number of test instances should not exceed number of samples in test set';
    error(msg);
end

% Assigning weights
% Computing the variance of each feature and assigning a weight to that
% feature depnding on normalized magnitude of variance
if weighting
    fprintf('Performing variance-based weighting');
    x_var = var(x_train,1);
    w = x_var./max(x_var);
else
    w = ones(1,size(x_train,2));
end

% Randomly subsample test data
rand_ind = randsample(size(x_test,1),n_test_instances);
x_test = x_test(rand_ind,:);
y_classes = y_classes(rand_ind);
n_test_data = size(x_test,1);

% Vector for holding guessed classes
y_hat = zeros(n_test_data,1);

accuracy = 0;

% Step through test data
for i=1:size(x_test,1)
    fprintf('\nTest Sample %i of %i\n',i,n_test_data);
    knn_temp = zeros(n_train_data,1);
    
    % Step through training data, computing euclidean distance
    for j=1:length(x_train)
        knn_temp(j)=sqrt(sum((w.*x_test(i,:)-w.*x_train(j,:)).^2));
    end
    
    % Sort and extract top k entries
    [~,index]=sort(knn_temp);
    
    % Calculate votes and output class estimate
    [~,y_hat(i)] = max(sum(y_train(index(1:k),:),1),[],2); 
    fprintf('Guess: %i   |   Actual: %i\n',y_hat(i),y_classes(i));
    
    % Compute and print accuracy
    accuracy = accuracy + (y_hat(i)==y_classes(i));
    fprintf('Accuracy: %f\n',100*accuracy/i);
end

labels = class_to_output(y_hat);

% Print overall accuracy
fprintf('Accuracy: %f pct\n',100.*accuracy/n_test_data);

end
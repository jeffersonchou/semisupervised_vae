###
'''
Borrowed from original implementation: https://github.com/dpkingma/nips14-ssl (anglepy)
'''
###

import numpy as np
import pickle, gzip
from data.helpers import *
import os


def load_numpy(path, binarize_y=False):
    # MNIST dataset
    if os.getcwd() not in path: path = os.getcwd() + '/' + path
    f = gzip.open(path, 'rb')
    train, valid, test = pickle.load(f, encoding='latin1')
    f.close()
    train_x, train_y = train
    valid_x, valid_y = valid
    test_x, test_y = test
    if binarize_y:
        train_y = binarize_labels(train_y)
        valid_y = binarize_labels(valid_y)
        test_y = binarize_labels(test_y)
        
    return train_x.T, train_y, valid_x.T, valid_y, test_x.T, test_y

# Loads data where data is split into class labels
def load_numpy_split(binarize_y=False, n_train=50000):
    path='mnist/mnist_28.pkl.gz'
    path = os.getcwd() + '/' + path
    train_x, train_y, valid_x, valid_y, test_x, test_y = load_numpy(path,False)

    train_x = train_x[0:n_train]
    train_y = train_y[0:n_train]
    
    def split_by_class(x, y, num_classes):
        result_x = [0]*num_classes
        result_y = [0]*num_classes
        for i in range(num_classes):
            idx_i = np.where(y == i)[0]
            result_x[i] = x[:,idx_i]
            result_y[i] = y[idx_i]
        return result_x, result_y
    
    train_x, train_y = split_by_class(train_x, train_y, 10)
    if binarize_y:
        valid_y = binarize_labels(valid_y)
        test_y = binarize_labels(test_y)
        for i in range(10):
            train_y[i] = binarize_labels(train_y[i])
    return train_x, train_y, valid_x, valid_y, test_x, test_y

def save_reshaped(shape):
    def reshape_digits(x, shape):
        def rebin(a, shape):
            sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
            return a.reshape(sh).mean(-1).mean(1)
        nrows = x.shape[0]
        ncols = shape[0]*shape[1]
        result = np.zeros((nrows, ncols))
        for i in range(nrows):
            result[i,:] = rebin(x[i,:].reshape((28,28)), shape).reshape((1, ncols))
        return result

    # MNIST dataset
    f = gzip.open(paths[28], 'rb')
    train, valid, test = pickle.load(f, encoding='latin1')
    train = reshape_digits(train[0], shape), train[1]
    valid = reshape_digits(valid[0], shape), valid[1]
    test = reshape_digits(test[0], shape), test[1]
    f.close()
    f = gzip.open(os.path.dirname(__file__)+'/mnist_'+str(shape[0])+'_.pkl.gz','wb')
    pickle.dump((train, valid, test), f)
    f.close()
 
       

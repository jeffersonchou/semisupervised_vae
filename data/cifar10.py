import h5py
import numpy as np

from data.helpers import *

def flatten(x):
    n = len(x)
    return np.reshape(x, (n, -1))


def load_numpy(path, binarize_y=False):
    data = h5py.File(path, 'r')
    train_x = flatten(data['X_train'])
    train_y = data['Y_train']
    valid_x = flatten(data['X_test'])
    valid_y = data['Y_test']
    ntest = valid_y.shape[0]
    test_x = valid_x[ntest//2:]
    test_y = valid_y[ntest//2:]
    valid_x = valid_x[:ntest//2]
    valid_y = valid_y[:ntest//2]

    if binarize_y:
        train_y = binarize_labels(train_y)
        valid_y = binarize_labels(valid_y)
        test_y = binarize_labels(test_y)
        
    return train_x.T, train_y, valid_x.T, valid_y, test_x.T, test_y

# Loads data where data is split into class labels
def load_numpy_split(binarize_y=False, n_train=50000):
    path='cifar10/cifar10.hdf5'
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



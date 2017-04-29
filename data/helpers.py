import numpy as np

def flatten(x):
    n=len(x)
    return np.reshape(x, (n, -1))

# Converts integer labels to binarized labels (1-of-K coding)
def binarize_labels(y, n_classes=10):
    new_y = np.zeros((n_classes, y.shape[0]))
    for i in range(y.shape[0]):
        new_y[y[i], i] = 1
    return new_y

def unbinarize_labels(y):
    return np.argmax(y,axis=0)
    
   
def make_random_projection(shape):
    W = np.random.uniform(low=-1, high=1, size=shape)
    W /= (np.sum(W**2,axis=1)**(1./2)).reshape((shape[0],1))
    return W


# Create semi-supervised sets of labeled and unlabeled data
# where there are equal number of labels from each class
# 'x': MNIST images
# 'y': MNIST labels (binarized / 1-of-K coded)
def create_semisupervised(x, y, n_ratio):
    import random
    n_x = sum([ _.shape[-1] for _ in x])
    n_classes = y[0].shape[0]
    n_labels_per_class = int((n_x * n_ratio)//n_classes)
    n_labeled=int(n_labels_per_class*n_classes)
    print('Total examples: {}, using {}/{} labels, ({} per class), (~{}% of data)'\
          .format(n_x, n_labeled, n_x, n_labels_per_class,
                  n_ratio*100))
    #if n_labeled%n_classes != 0: raise("n_labeled (wished number of labeled samples) not divisible by n_classes (number of classes)")
    x_labeled = [0]*n_classes
    x_unlabeled = [0]*n_classes
    y_labeled = [0]*n_classes
    y_unlabeled = [0]*n_classes
    for i in range(n_classes):
        idx = list(range(x[i].shape[1]))
        random.shuffle(idx)
        x_labeled[i] = x[i][:,idx[:n_labels_per_class]]
        y_labeled[i] = y[i][:,idx[:n_labels_per_class]]
        x_unlabeled[i] = x[i][:,idx[n_labels_per_class:]]
        y_unlabeled[i] = y[i][:,idx[n_labels_per_class:]]
    return np.hstack(x_labeled), np.hstack(y_labeled), np.hstack(x_unlabeled), np.hstack(y_unlabeled), n_labeled

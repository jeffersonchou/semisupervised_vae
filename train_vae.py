from vae import VariationalAutoencoder
import numpy as np
#import data.mnist as mnist #https://github.com/dpkingma/nips14-ssl
from data import mnist, cifar10, svhn
import argparse


def main(flags, data):

    #############################
    ''' Experiment Parameters '''
    #############################

    num_batches = flags.num_batches
    dim_z = flags.dim_z
    epochs = flags.epochs
    learning_rate = flags.lr
    l2_loss = flags.l2_loss
    seed = flags.seed
    data_path=flags.datapath

    #Neural Networks parameterising p(x|z), q(z|x)
    hidden_layers_px = [ 600, 600 ]
    hidden_layers_qz = [ 600, 600 ]

    ####################
    ''' Load Dataset '''
    ####################

    #Uses anglpy module from original paper (linked at top) to load the dataset
    train_x, train_y, valid_x, valid_y, test_x, test_y\
            = data.load_numpy(data_path, binarize_y=True)

    x_train, y_train = train_x.T, train_y.T
    x_valid, y_valid = valid_x.T, valid_y.T
    x_test, y_test = test_x.T, test_y.T

    dim_x = x_train.shape[1]
    dim_y = y_train.shape[1]

    ######################################
    ''' Train Variational Auto-Encoder '''
    ######################################

    VAE = VariationalAutoencoder(dim_x = dim_x, 
                                 dim_z = dim_z,
                                 hidden_layers_px = hidden_layers_px,
                                 hidden_layers_qz = hidden_layers_qz,
                                 l2_loss = l2_loss )

    #draw_img uses pylab and seaborn to draw images of original vs. reconstruction 
    #every n iterations (set to 0 to disable)

    VAE.train(x = x_train, x_valid = x_valid, 
              epochs = epochs, num_batches = num_batches,
              learning_rate = learning_rate, seed = seed, 
              stop_iter = 30, print_every = 10, draw_img = 0,
              save_path = flags.vaemodel)

if __name__ == '__main__':
    num_batches = 1000      #Number of minibatches in a single epoch
    dim_z = 50              #Dimensionality of latent variable (z)
    epochs = 3001           #Number of epochs through the full dataset
    learning_rate = 3e-4    #Learning rate of ADAM
    l2_loss = 1e-6          #L2 Regularisation weight
    seed = 31415            #Seed for RNG
    dataset = 'mnist'
    mnist_path= 'mnist/mnist_28.pkl.gz'
    datasets={'mnist':mnist, 'cifar':cifar10, 'svhn':svhn}

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default=dataset, type=str,
                        choices=['mnist', 'cifar', 'svhn'],
                        help='Dataset to use, currently has mnist(def) and cifar10(TODO)')
    parser.add_argument('--vaemodel', default='models/VAE_{}.cpkt'.format(dataset), 
                        type=str,
                        help='VAE model to load')
    parser.add_argument('--num_batches', default=num_batches, type=int,
                        help="Number of minibatches in each epoch")
    parser.add_argument('--dim_z', default=dim_z, type=int,
                        help='Dimension of latent space z')
    parser.add_argument('--epochs', default=epochs, type=int,
                        help="Number of epochs to train")
    parser.add_argument('--lr', default=learning_rate, type=float,
                        help='Adam learning rate')
    parser.add_argument('--l2_loss', default=1e-6, type=float,
                        help='L2 Regularization weight')
    parser.add_argument('--seed', default=seed, type=int)
    parser.add_argument('--datapath', default=mnist_path, type=str,
                        help='Path to data')
    flags= parser.parse_args()
    data = datasets[flags.dataset]

    main(flags, data)


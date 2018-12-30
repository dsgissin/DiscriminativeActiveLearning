"""
The main file which runs our active learning experiments. The experiment results are saved in pickle files that we later
analyze over many experiments to produce the plots in our blog.
"""

import pickle
import os
import sys
import argparse
from keras.utils import to_categorical
from sklearn.datasets import load_boston, load_diabetes

from models import *
from query_methods import *

def parse_input():
    p = argparse.ArgumentParser()
    p.add_argument('experiment_index', type=int, help="index of current experiment")
    p.add_argument('data_type', type=str, choices={'mnist', 'cifar10', 'cifar100'}, help="data type (mnist/cifar10/cifar100)")
    p.add_argument('batch_size', type=int, help="active learning batch size")
    p.add_argument('initial_size', type=int, help="initial sample size for active learning")
    p.add_argument('iterations', type=int, help="number of active learning batches to sample")
    p.add_argument('method', type=str,
                   choices={'Random','CoreSet','CoreSetMIP','Discriminative','DiscriminativeLearned','DiscriminativeAE','DiscriminativeStochastic','Uncertainty','Bayesian','UncertaintyEntropy','BayesianEntropy','EGL','Adversarial'},
                   help="sampling method ('Random','CoreSet','CoreSetMIP','Discriminative','DiscriminativeLearned','DiscriminativeAE','DiscriminativeStochastic','Uncertainty','Bayesian','UncertaintyEntropy','BayesianEntropy','EGL','Adversarial')")
    p.add_argument('experiment_folder', type=str,
                   help="folder where the experiment results will be saved")
    p.add_argument('--method2', '-method2', type=str,
                   choices={None,'Random','CoreSet','CoreSetMIP','Discriminative','DiscriminativeLearned','DiscriminativeAE','DiscriminativeStochastic','Uncertainty','Bayesian','UncertaintyEntropy','BayesianEntropy','EGL','Adversarial'},
                   default=None,
                   help="second sampling method ('Random','CoreSet','CoreSetMIP','Discriminative','DiscriminativeLearned','DiscriminativeAE','DiscriminativeStochastic','Uncertainty','Bayesian','UncertaintyEntropy','BayesianEntropy','EGL','Adversarial')")
    p.add_argument('--initial_idx_path', '-idx', type=str,
                   default=None,
                   help="path to a folder with a pickle file with the initial indices of the labeled set")
    p.add_argument('--gpu', '-gpu', type=int, default=2)
    args = p.parse_args()
    return args


def load_batch(fpath, label_key='labels'):

    with open(fpath, 'rb') as f:
        if sys.version_info < (3,):
            d = pickle.load(f)
        else:
            d = pickle.load(f, encoding='bytes')
            # decode utf8
            d_decoded = {}
            for k, v in d.items():
                d_decoded[k.decode('utf8')] = v
            d = d_decoded
    data = d['data']
    labels = d[label_key]

    data = data.reshape(data.shape[0], 3, 32, 32)
    return data, labels


def load_mnist():
    """
    load and pre-process the MNIST data
    """

    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if K.image_data_format() == 'channels_last':
        x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
        x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))
    else:
        x_train = x_train.reshape((x_train.shape[0], 1, 28, 28))
        x_test = x_test.reshape((x_test.shape[0], 1, 28, 28))

    # standardise the dataset:
    x_train = np.array(x_train).astype('float32') / 255
    x_test = np.array(x_test).astype('float32') / 255

    # shuffle the data:
    perm = np.random.permutation(x_train.shape[0])
    x_train = x_train[perm]
    y_train = y_train[perm]

    return (x_train, y_train), (x_test, y_test)


def load_cifar_10():
    """
    load and pre-process the CIFAR-10 data
    """

    dirname = ''  # TODO: your path here

    num_train_samples = 50000

    x_train = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
    y_train = np.empty((num_train_samples,), dtype='uint8')

    for i in range(1, 6):
        fpath = os.path.join(dirname, 'data_batch_' + str(i))
        (x_train[(i - 1) * 10000: i * 10000, :, :, :],
         y_train[(i - 1) * 10000: i * 10000]) = load_batch(fpath)

    fpath = os.path.join(dirname, 'test_batch')
    x_test, y_test = load_batch(fpath)

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    # standardise the dataset:
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    if K.image_data_format() == 'channels_last':
        x_train = x_train.transpose(0, 2, 3, 1)
        x_test = x_test.transpose(0, 2, 3, 1)

    # shuffle the data:
    perm = np.random.permutation(x_train.shape[0])
    x_train = x_train[perm]
    y_train = y_train[perm]

    return (x_train, y_train), (x_test, y_test)


def load_cifar_100(label_mode='fine'):
    """
    load and pre-process the CIFAR-100 data
    """

    dirname = ''  # TODO: your path here

    fpath = os.path.join(dirname, 'train')
    x_train, y_train = load_batch(fpath, label_key=label_mode + '_labels')

    fpath = os.path.join(dirname, 'test')
    x_test, y_test = load_batch(fpath, label_key=label_mode + '_labels')

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    # standardise the dataset:
    x_train = np.array(x_train).astype('float32') / 255
    x_test = np.array(x_test).astype('float32') / 255

    if K.image_data_format() == 'channels_last':
        x_train = x_train.transpose(0, 2, 3, 1)
        x_test = x_test.transpose(0, 2, 3, 1)

    # shuffle the data:
    perm = np.random.permutation(x_train.shape[0])
    x_train = x_train[perm]
    y_train = y_train[perm]

    return (x_train, y_train), (x_test, y_test)


def evaluate_sample(training_function, X_train, Y_train, X_test, Y_test, checkpoint_path):
    """
    A function that accepts a labeled-unlabeled data split and trains the relevant model on the labeled data, returning
    the model and it's accuracy on the test set.
    """

    # shuffle the training set:
    perm = np.random.permutation(X_train.shape[0])
    X_train = X_train[perm]
    Y_train = Y_train[perm]

    # create the validation set:
    X_validation = X_train[:int(0.2*X_train.shape[0])]
    Y_validation = Y_train[:int(0.2*Y_train.shape[0])]
    X_train = X_train[int(0.2*X_train.shape[0]):]
    Y_train = Y_train[int(0.2*Y_train.shape[0]):]

    # train and evaluate the model:
    model = training_function(X_train, Y_train, X_validation, Y_validation, checkpoint_path, gpu=args.gpu)
    if args.data_type in ['imdb', 'wiki']:
        acc = model.evaluate(X_test, Y_test, verbose=0)
    else:
        loss, acc = model.evaluate(X_test, Y_test, verbose=0)

    return acc, model


if __name__ == '__main__':

    # parse the arguments:
    args = parse_input()

    # load the dataset:
    if args.data_type == 'mnist':
        (X_train, Y_train), (X_test, Y_test) = load_mnist()
        num_labels = 10
        if K.image_data_format() == 'channels_last':
            input_shape = (28, 28, 1)
        else:
            input_shape = (1, 28, 28)
        evaluation_function = train_mnist_model
    if args.data_type == 'cifar10':
        (X_train, Y_train), (X_test, Y_test) = load_cifar_10()
        num_labels = 10
        if K.image_data_format() == 'channels_last':
            input_shape = (32, 32, 3)
        else:
            input_shape = (3, 32, 32)
        evaluation_function = train_cifar10_model
    if args.data_type == 'cifar100':
        (X_train, Y_train), (X_test, Y_test) = load_cifar_100()
        num_labels = 100
        if K.image_data_format() == 'channels_last':
            input_shape = (32,32,3)
        else:
            input_shape = (3, 32, 32)
        evaluation_function = train_cifar100_model

    # make categorical:
    Y_train = to_categorical(Y_train)
    Y_test = to_categorical(Y_test)

    # load the indices:
    if args.initial_idx_path is not None:
        idx_path = os.path.join(args.initial_idx_path, '{exp}_{size}_{data}.pkl'.format(exp=args.experiment_index, size=args.initial_size, data=args.data_type))
        with open(idx_path, 'rb') as f:
            labeled_idx = pickle.load(f)
    else:
        print("No Initial Indices Found - Drawing Random Indices...")
        labeled_idx = np.random.choice(X_train.shape[0], args.initial_size, replace=False)

    # set the first query method:
    if args.method == 'Random':
        method = RandomSampling
    elif args.method == 'CoreSet':
        method = CoreSetSampling
    elif args.method == 'CoreSetMIP':
        method = CoreSetMIPSampling
    elif args.method == 'Discriminative':
        method = DiscriminativeSampling
    elif args.method == 'DiscriminativeLearned':
        method = DiscriminativeRepresentationSampling
    elif args.method == 'DiscriminativeAE':
        method = DiscriminativeAutoencoderSampling
    elif args.method == 'DiscriminativeStochastic':
        method = DiscriminativeStochasticSampling
    elif args.method == 'Uncertainty':
        method = UncertaintySampling
    elif args.method == 'Bayesian':
        method = BayesianUncertaintySampling
    elif args.method == 'UncertaintyEntropy':
        method = UncertaintyEntropySampling
    elif args.method == 'BayesianEntropy':
        method = BayesianUncertaintyEntropySampling
    elif args.method == 'EGL':
        method = EGLSampling
    elif args.method == 'Adversarial':
        method = AdversarialSampling

    # set the second query method:
    if args.method2 is not None:
        print("Using Two Methods...")
        if args.method2 == 'Random':
            method2 = RandomSampling
        elif args.method2 == 'CoreSet':
            method2 = CoreSetSampling
        elif args.method2 == 'CoreSetMIP':
            method2 = CoreSetMIPSampling
        elif args.method2 == 'Discriminative':
            method2 = DiscriminativeSampling
        elif args.method2 == 'DiscriminativeLearned':
            method2 = DiscriminativeRepresentationSampling
        elif args.method2 == 'DiscriminativeAE':
            method2 = DiscriminativeAutoencoderSampling
        elif args.method2 == 'DiscriminativeStochastic':
            method2 = DiscriminativeStochasticSampling
        elif args.method2 == 'Uncertainty':
            method2 = UncertaintySampling
        elif args.method2 == 'Bayesian':
            method2 = BayesianUncertaintySampling
        elif args.method2 == 'UncertaintyEntropy':
            method2 = UncertaintyEntropySampling
        elif args.method2 == 'BayesianEntropy':
            method2 = BayesianUncertaintyEntropySampling
        elif args.method2 == 'EGL':
            method2 = EGLSampling
        elif args.method2 == 'Adversarial':
            method2 = AdversarialSampling
        else:
            print("ERROR - UNKNOWN SECOND METHOD!")
            exit()
    else:
        method2 = None
        print("Only One Method Used...")

    # create the QueryMethod object:
    if method2 is not None:
        query_method = CombinedSampling(None, input_shape, num_labels, method, method2, args.gpu)
    else:
        query_method = method(None, input_shape, num_labels, args.gpu)

    # create the checkpoint path:
    if not os.path.isdir(os.path.join(args.experiment_folder, 'models')):
        os.mkdir(os.path.join(args.experiment_folder, 'models'))
    model_folder = os.path.join(args.experiment_folder, 'models')
    if method2 is None:
        checkpoint_path = os.path.join(model_folder, '{alg}_{datatype}_{init}_{batch_size}_{idx}.hdf5'.format(
            alg=args.method, datatype=args.data_type, batch_size=args.batch_size, init=args.initial_size, idx=args.experiment_index
        ))
    else:
        checkpoint_path = os.path.join(model_folder, '{alg}_{alg2}_{datatype}_{init}_{batch_size}_{idx}.hdf5'.format(
            alg=args.method, alg2=args.method2, datatype=args.data_type, batch_size=args.batch_size, init=args.initial_size, idx=args.experiment_index
        ))

    # create the results path:
    if not os.path.isdir(os.path.join(args.experiment_folder, 'results')):
        os.mkdir(os.path.join(args.experiment_folder, 'results'))
    results_folder = os.path.join(args.experiment_folder, 'results')
    if method2 is None:
        results_path = os.path.join(results_folder, '{alg}_{datatype}_{init}_{batch_size}_{idx}.pkl'.format(
            alg=args.method, datatype=args.data_type, batch_size=args.batch_size, init=args.initial_size, idx=args.experiment_index
        ))
    else:
        results_path = os.path.join(results_folder, '{alg}_{alg2}_{datatype}_{init}_{batch_size}_{idx}.pkl'.format(
            alg=args.method, alg2=args.method2, datatype=args.data_type, batch_size=args.batch_size, init=args.initial_size, idx=args.experiment_index
        ))

    # create the label entropy path:
    if method2 is None:
        entropy_path = os.path.join(results_folder, '{alg}_{datatype}_{init}_{batch_size}_{idx}_entropy.pkl'.format(
            alg=args.method, datatype=args.data_type, batch_size=args.batch_size, init=args.initial_size, idx=args.experiment_index
        ))
    else:
        entropy_path = os.path.join(results_folder, '{alg}_{alg2}_{datatype}_{init}_{batch_size}_{idx}_entropy.pkl'.format(
            alg=args.method, alg2=args.method2, datatype=args.data_type, batch_size=args.batch_size, init=args.initial_size, idx=args.experiment_index
        ))

    # run the experiment:
    accuracies = []
    entropies = []
    label_distributions = []
    queries = []
    acc, model = evaluate_sample(evaluation_function, X_train[labeled_idx,:], Y_train[labeled_idx], X_test, Y_test, checkpoint_path)
    query_method.update_model(model)
    accuracies.append(acc)
    print("Test Accuracy Is " + str(acc))
    for i in range(args.iterations):

        # get the new indices from the algorithm
        old_labeled = np.copy(labeled_idx)
        labeled_idx = query_method.query(X_train, Y_train, labeled_idx, args.batch_size)

        # calculate and store the label entropy:
        new_idx = labeled_idx[np.logical_not(np.isin(labeled_idx, old_labeled))]
        new_labels = Y_train[new_idx]
        new_labels /= np.sum(new_labels)
        new_labels = np.sum(new_labels, axis=0)
        entropy = -np.sum(new_labels * np.log(new_labels + 1e-10))
        entropies.append(entropy)
        label_distributions.append(new_labels)
        queries.append(new_idx)

        # evaluate the new sample:
        acc, model = evaluate_sample(evaluation_function, X_train[labeled_idx], Y_train[labeled_idx], X_test, Y_test, checkpoint_path)
        query_method.update_model(model)
        accuracies.append(acc)
        print("Test Accuracy Is " + str(acc))

    # save the results:
    with open(results_path, 'wb') as f:
        pickle.dump([accuracies, args.initial_size, args.batch_size], f)
        print("Saved results to " + results_path)
    with open(entropy_path, 'wb') as f:
        pickle.dump([entropies, label_distributions, queries], f)
        print("Saved entropy statistics to " + entropy_path)

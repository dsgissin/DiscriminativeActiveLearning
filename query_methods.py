"""
The file containing implementations to all of the query strategies. References to all of these methods can be found in
the blog that accompanies this code.
"""

import gc
from scipy.spatial import distance_matrix

from keras.models import Model
import keras.backend as K
from keras.losses import categorical_crossentropy
from keras.layers import Lambda
from keras import optimizers
from cleverhans.attacks import FastGradientMethod, DeepFool
from cleverhans.utils_keras import KerasModelWrapper

from models import *

def get_unlabeled_idx(X_train, labeled_idx):
    """
    Given the training set and the indices of the labeled examples, return the indices of the unlabeled examples.
    """
    return np.arange(X_train.shape[0])[np.logical_not(np.in1d(np.arange(X_train.shape[0]), labeled_idx))]


class QueryMethod:
    """
    A general class for query strategies, with a general method for querying examples to be labeled.
    """

    def __init__(self, model, input_shape=(28,28), num_labels=10, gpu=1):
        self.model = model
        self.input_shape = input_shape
        self.num_labels = num_labels
        self.gpu = gpu

    def query(self, X_train, Y_train, labeled_idx, amount):
        """
        get the indices of labeled examples after the given amount have been queried by the query strategy.
        :param X_train: the training set
        :param Y_train: the training labels
        :param labeled_idx: the indices of the labeled examples
        :param amount: the amount of examples to query
        :return: the new labeled indices (including the ones queried)
        """
        return NotImplemented

    def update_model(self, new_model):
        del self.model
        gc.collect()
        self.model = new_model


class RandomSampling(QueryMethod):
    """
    A random sampling query strategy baseline.
    """

    def __init__(self, model, input_shape, num_labels, gpu):
        super().__init__(model, input_shape, num_labels, gpu)

    def query(self, X_train, Y_train, labeled_idx, amount):
        unlabeled_idx = get_unlabeled_idx(X_train, labeled_idx)
        return np.hstack((labeled_idx, np.random.choice(unlabeled_idx, amount, replace=False)))


class UncertaintySampling(QueryMethod):
    """
    The basic uncertainty sampling query strategy, querying the examples with the minimal top confidence.
    """

    def __init__(self, model, input_shape, num_labels, gpu):
        super().__init__(model, input_shape, num_labels, gpu)

    def query(self, X_train, Y_train, labeled_idx, amount):

        unlabeled_idx = get_unlabeled_idx(X_train, labeled_idx)
        predictions = self.model.predict(X_train[unlabeled_idx, :])

        unlabeled_predictions = np.amax(predictions, axis=1)

        selected_indices = np.argpartition(unlabeled_predictions, amount)[:amount]
        return np.hstack((labeled_idx, unlabeled_idx[selected_indices]))


class UncertaintyEntropySampling(QueryMethod):
    """
    The basic uncertainty sampling query strategy, querying the examples with the top entropy.
    """

    def __init__(self, model, input_shape, num_labels, gpu):
        super().__init__(model, input_shape, num_labels, gpu)

    def query(self, X_train, Y_train, labeled_idx, amount):

        unlabeled_idx = get_unlabeled_idx(X_train, labeled_idx)
        predictions = self.model.predict(X_train[unlabeled_idx, :])

        unlabeled_predictions = np.sum(predictions * np.log(predictions + 1e-10), axis=1)

        selected_indices = np.argpartition(unlabeled_predictions, amount)[:amount]
        return np.hstack((labeled_idx, unlabeled_idx[selected_indices]))


class BayesianUncertaintySampling(QueryMethod):
    """
    An implementation of the Bayesian active learning method, using minimal top confidence as the decision rule.
    """

    def __init__(self, model, input_shape, num_labels, gpu):
        super().__init__(model, input_shape, num_labels, gpu)

        self.T = 20

    def dropout_predict(self, data):

        f = K.function([self.model.layers[0].input, K.learning_phase()],
                       [self.model.layers[-1].output])
        predictions = np.zeros((self.T, data.shape[0], self.num_labels))
        for t in range(self.T):
            predictions[t,:,:] = f([data, 1])[0]

        final_prediction = np.mean(predictions, axis=0)
        prediction_uncertainty = np.std(predictions, axis=0)

        return final_prediction, prediction_uncertainty

    def query(self, X_train, Y_train, labeled_idx, amount):

        unlabeled_idx = get_unlabeled_idx(X_train, labeled_idx)

        predictions = np.zeros((unlabeled_idx.shape[0], self.num_labels))
        uncertainties = np.zeros((unlabeled_idx.shape[0], self.num_labels))
        i = 0
        split = 128  # split into iterations of 128 due to memory constraints
        while i < unlabeled_idx.shape[0]:

            if i+split > unlabeled_idx.shape[0]:
                preds, unc = self.dropout_predict(X_train[unlabeled_idx[i:], :])
                predictions[i:] = preds
                uncertainties[i:] = unc
            else:
                preds, unc = self.dropout_predict(X_train[unlabeled_idx[i:i+split], :])
                predictions[i:i+split] = preds
                uncertainties[i:i+split] = unc
            i += split

        unlabeled_predictions = np.amax(predictions, axis=1)
        selected_indices = np.argpartition(unlabeled_predictions, amount)[:amount]
        return np.hstack((labeled_idx, unlabeled_idx[selected_indices]))


class BayesianUncertaintyEntropySampling(QueryMethod):
    """
    An implementation of the Bayesian active learning method, using maximal entropy as the decision rule.
    """

    def __init__(self, model, input_shape, num_labels, gpu):
        super().__init__(model, input_shape, num_labels, gpu)

        self.T = 100

    def dropout_predict(self, data):

        f = K.function([self.model.layers[0].input, K.learning_phase()],
                       [self.model.layers[-1].output])
        predictions = np.zeros((self.T, data.shape[0], self.num_labels))
        for t in range(self.T):
            predictions[t,:,:] = f([data, 1])[0]

        final_prediction = np.mean(predictions, axis=0)
        prediction_uncertainty = np.std(predictions, axis=0)

        return final_prediction, prediction_uncertainty

    def query(self, X_train, Y_train, labeled_idx, amount):

        unlabeled_idx = get_unlabeled_idx(X_train, labeled_idx)

        predictions = np.zeros((unlabeled_idx.shape[0], self.num_labels))
        i = 0
        while i < unlabeled_idx.shape[0]: # split into iterations of 1000 due to memory constraints

            if i+1000 > unlabeled_idx.shape[0]:
                preds, _ = self.dropout_predict(X_train[unlabeled_idx[i:], :])
                predictions[i:] = preds
            else:
                preds, _ = self.dropout_predict(X_train[unlabeled_idx[i:i+1000], :])
                predictions[i:i+1000] = preds

            i += 1000

        unlabeled_predictions = np.sum(predictions * np.log(predictions + 1e-10), axis=1)
        selected_indices = np.argpartition(unlabeled_predictions, amount)[:amount]
        return np.hstack((labeled_idx, unlabeled_idx[selected_indices]))


class AdversarialSampling(QueryMethod):
    """
    An implementation of adversarial active learning, using cleverhans' implementation of DeepFool to generate
    adversarial examples.
    """

    def __init__(self, model, input_shape, num_labels, gpu):
        super().__init__(model, input_shape, num_labels, gpu)

    def query(self, X_train, Y_train, labeled_idx, amount):

        unlabeled_idx = get_unlabeled_idx(X_train, labeled_idx)
        unlabeled = X_train[unlabeled_idx]

        keras_wrapper = KerasModelWrapper(self.model)
        sess = K.get_session()
        deep_fool = DeepFool(keras_wrapper, sess=sess)
        deep_fool_params = {'over_shoot': 0.02,
                            'clip_min': 0.,
                            'clip_max': 1.,
                            'nb_candidate': Y_train.shape[1],
                            'max_iter': 10}
        true_predictions = np.argmax(self.model.predict(unlabeled, batch_size=256), axis=1)
        adversarial_predictions = np.copy(true_predictions)
        while np.sum(true_predictions != adversarial_predictions) < amount:
            adversarial_images = np.zeros(unlabeled.shape)
            for i in range(0, unlabeled.shape[0], 100):
                print("At {i} out of {n}".format(i=i, n=unlabeled.shape[0]))
                if i+100 > unlabeled.shape[0]:
                    adversarial_images[i:] = deep_fool.generate_np(unlabeled[i:], **deep_fool_params)
                else:
                    adversarial_images[i:i+100] = deep_fool.generate_np(unlabeled[i:i+100], **deep_fool_params)
            pertubations = adversarial_images - unlabeled
            norms = np.linalg.norm(np.reshape(pertubations,(unlabeled.shape[0],-1)), axis=1)
            adversarial_predictions = np.argmax(self.model.predict(adversarial_images, batch_size=256), axis=1)
            norms[true_predictions == adversarial_predictions] = np.inf
            deep_fool_params['max_iter'] *= 2

        selected_indices = np.argpartition(norms, amount)[:amount]

        del keras_wrapper
        del deep_fool
        gc.collect()

        return np.hstack((labeled_idx, unlabeled_idx[selected_indices]))


class DiscriminativeSampling(QueryMethod):
    """
    An implementation of DAL (discriminative active learning), using the raw pixels as the representation.
    """

    def __init__(self, model, input_shape, num_labels, gpu):
        super().__init__(model, input_shape, num_labels, gpu)

        self.sub_batches = 10

    def query(self, X_train, Y_train, labeled_idx, amount):

        # subsample from the unlabeled set:
        unlabeled_idx = get_unlabeled_idx(X_train, labeled_idx)
        unlabeled_idx = np.random.choice(unlabeled_idx, np.min([labeled_idx.shape[0]*10, unlabeled_idx.size]), replace=False)

        # iteratively sub-sample using the discriminative sampling routine:
        labeled_so_far = 0
        sub_sample_size = int(amount / self.sub_batches)
        while labeled_so_far < amount:
            if labeled_so_far + sub_sample_size > amount:
                sub_sample_size = amount - labeled_so_far

            model = train_discriminative_model(X_train[labeled_idx], X_train[unlabeled_idx], self.input_shape, gpu=self.gpu)
            predictions = model.predict(X_train[unlabeled_idx])
            selected_indices = np.argpartition(predictions[:,1], -sub_sample_size)[-sub_sample_size:]
            labeled_idx = np.hstack((labeled_idx, unlabeled_idx[selected_indices]))
            labeled_so_far += sub_sample_size
            unlabeled_idx = get_unlabeled_idx(X_train, labeled_idx)
            unlabeled_idx = np.random.choice(unlabeled_idx, np.min([labeled_idx.shape[0]*10, unlabeled_idx.size]), replace=False)

            # delete the model to free GPU memory:
            del model
            gc.collect()

        return labeled_idx


class DiscriminativeRepresentationSampling(QueryMethod):
    """
    An implementation of DAL (discriminative active learning), using the learned representation as our representation.
    This implementation is the one which performs best in practice.
    """

    def __init__(self, model, input_shape, num_labels, gpu):
        super().__init__(model, input_shape, num_labels, gpu)

        self.sub_batches = 20


    def query(self, X_train, Y_train, labeled_idx, amount):

        # subsample from the unlabeled set:
        unlabeled_idx = get_unlabeled_idx(X_train, labeled_idx)
        unlabeled_idx = np.random.choice(unlabeled_idx, np.min([labeled_idx.shape[0]*10, unlabeled_idx.size]), replace=False)

        embedding_model = Model(inputs=self.model.input,
                                outputs=self.model.get_layer('softmax').input)
        representation = embedding_model.predict(X_train, batch_size=128).reshape((X_train.shape[0], -1, 1))

        # iteratively sub-sample using the discriminative sampling routine:
        labeled_so_far = 0
        sub_sample_size = int(amount / self.sub_batches)
        while labeled_so_far < amount:
            if labeled_so_far + sub_sample_size > amount:
                sub_sample_size = amount - labeled_so_far

            model = train_discriminative_model(representation[labeled_idx], representation[unlabeled_idx], representation[0].shape, gpu=self.gpu)
            predictions = model.predict(representation[unlabeled_idx])
            selected_indices = np.argpartition(predictions[:,1], -sub_sample_size)[-sub_sample_size:]
            labeled_idx = np.hstack((labeled_idx, unlabeled_idx[selected_indices]))
            labeled_so_far += sub_sample_size
            unlabeled_idx = get_unlabeled_idx(X_train, labeled_idx)
            unlabeled_idx = np.random.choice(unlabeled_idx, np.min([labeled_idx.shape[0]*10, unlabeled_idx.size]), replace=False)

            # delete the model to free GPU memory:
            del model
            gc.collect()
        del embedding_model
        gc.collect()

        return labeled_idx


class DiscriminativeAutoencoderSampling(QueryMethod):
    """
    An implementation of DAL (discriminative active learning), using an autoencoder embedding as our representation.
    """

    def __init__(self, model, input_shape, num_labels, gpu):
        super().__init__(model, input_shape, num_labels, gpu)

        self.sub_batches = 10
        self.autoencoder = None
        self.embedding = None

    def query(self, X_train, Y_train, labeled_idx, amount):

        if self.autoencoder is None:
            self.autoencoder = get_autoencoder_model(input_shape=(28,28,1))
            self.autoencoder.compile(optimizer=optimizers.Adam(lr=0.0003), loss='binary_crossentropy')
            self.autoencoder.fit(X_train, X_train,
                                 epochs=200,
                                 batch_size=256,
                                 shuffle=True,
                                 verbose=2)
            encoder = Model(self.autoencoder.input, self.autoencoder.get_layer('embedding').input)
            self.embedding = encoder.predict(X_train.reshape((-1,28,28,1)), batch_size=1024)

        # subsample from the unlabeled set:
        unlabeled_idx = get_unlabeled_idx(X_train, labeled_idx)
        unlabeled_idx = np.random.choice(unlabeled_idx, np.min([labeled_idx.shape[0]*10, unlabeled_idx.size]), replace=False)

        # iteratively sub-sample using the discriminative sampling routine:
        labeled_so_far = 0
        sub_sample_size = int(amount / self.sub_batches)
        while labeled_so_far < amount:
            if labeled_so_far + sub_sample_size > amount:
                sub_sample_size = amount - labeled_so_far

            model = train_discriminative_model(self.embedding[labeled_idx], self.embedding[unlabeled_idx], self.embedding[0].shape, gpu=self.gpu)
            predictions = model.predict(self.embedding[unlabeled_idx])
            selected_indices = np.argpartition(predictions[:,1], -sub_sample_size)[-sub_sample_size:]
            labeled_idx = np.hstack((labeled_idx, unlabeled_idx[selected_indices]))
            labeled_so_far += sub_sample_size
            unlabeled_idx = get_unlabeled_idx(X_train, labeled_idx)
            unlabeled_idx = np.random.choice(unlabeled_idx, np.min([labeled_idx.shape[0]*10, unlabeled_idx.size]), replace=False)

            # delete the model to free GPU memory:
            del model
            gc.collect()

        return labeled_idx


class DiscriminativeStochasticSampling(QueryMethod):
    """
    An implementation of DAL (discriminative active learning), using the learned representation as our representation
    and sampling proportionally to the confidence as being "unlabeled".
    """

    def __init__(self, model, input_shape, num_labels, gpu):
        super().__init__(model, input_shape, num_labels, gpu)

        self.sub_batches = 10
        self.temperature = 0.01

    def query(self, X_train, Y_train, labeled_idx, amount):

        # subsample from the unlabeled set:
        unlabeled_idx = get_unlabeled_idx(X_train, labeled_idx)
        unlabeled_idx = np.random.choice(unlabeled_idx, np.min([labeled_idx.shape[0]*10, unlabeled_idx.size]), replace=False)

        embedding_model = Model(inputs=self.model.input,
                                outputs=self.model.get_layer('softmax').input)
        representation = embedding_model.predict(X_train, batch_size=256).reshape((X_train.shape[0], -1, 1))

        # iteratively sub-sample using the discriminative sampling routine:
        labeled_so_far = 0
        sub_sample_size = int(amount / self.sub_batches)
        while labeled_so_far < amount:
            if labeled_so_far + sub_sample_size > amount:
                sub_sample_size = amount - labeled_so_far

            model = train_discriminative_model(representation[labeled_idx], representation[unlabeled_idx], representation[0].shape, gpu=self.gpu)
            predictions = model.predict(representation[unlabeled_idx])
            predictions -= 1  # for numerical stability
            predictions = np.exp(predictions / self.temperature)
            predictions[:,1] /= np.sum(predictions[:,1])
            selected_indices = np.random.choice(unlabeled_idx, sub_sample_size, replace=False, p=predictions[:,1])

            labeled_idx = np.hstack((labeled_idx, selected_indices))
            labeled_so_far += sub_sample_size
            unlabeled_idx = get_unlabeled_idx(X_train, labeled_idx)
            unlabeled_idx = np.random.choice(unlabeled_idx, np.min([labeled_idx.shape[0]*10, unlabeled_idx.size]), replace=False)

            # delete the model to free GPU memory:
            del model
            gc.collect()

        del embedding_model

        return labeled_idx


class CoreSetSampling(QueryMethod):
    """
    An implementation of the greedy core set query strategy.
    """

    def __init__(self, model, input_shape, num_labels, gpu):
        super().__init__(model, input_shape, num_labels, gpu)

    def greedy_k_center(self, labeled, unlabeled, amount):

        greedy_indices = []

        # get the minimum distances between the labeled and unlabeled examples (iteratively, to avoid memory issues):
        min_dist = np.min(distance_matrix(labeled[0, :].reshape((1, labeled.shape[1])), unlabeled), axis=0)
        min_dist = min_dist.reshape((1, min_dist.shape[0]))
        for j in range(1, labeled.shape[0], 100):
            if j + 100 < labeled.shape[0]:
                dist = distance_matrix(labeled[j:j+100, :], unlabeled)
            else:
                dist = distance_matrix(labeled[j:, :], unlabeled)
            min_dist = np.vstack((min_dist, np.min(dist, axis=0).reshape((1, min_dist.shape[1]))))
            min_dist = np.min(min_dist, axis=0)
            min_dist = min_dist.reshape((1, min_dist.shape[0]))

        # iteratively insert the farthest index and recalculate the minimum distances:
        farthest = np.argmax(min_dist)
        greedy_indices.append(farthest)
        for i in range(amount-1):
            dist = distance_matrix(unlabeled[greedy_indices[-1], :].reshape((1,unlabeled.shape[1])), unlabeled)
            min_dist = np.vstack((min_dist, dist.reshape((1, min_dist.shape[1]))))
            min_dist = np.min(min_dist, axis=0)
            min_dist = min_dist.reshape((1, min_dist.shape[0]))
            farthest = np.argmax(min_dist)
            greedy_indices.append(farthest)

        return np.array(greedy_indices)

    def query(self, X_train, Y_train, labeled_idx, amount):

        unlabeled_idx = get_unlabeled_idx(X_train, labeled_idx)

        # use the learned representation for the k-greedy-center algorithm:
        representation_model = Model(inputs=self.model.input, outputs=self.model.get_layer('softmax').input)
        representation = representation_model.predict(X_train, verbose=0)
        new_indices = self.greedy_k_center(representation[labeled_idx, :], representation[unlabeled_idx, :], amount)
        return np.hstack((labeled_idx, unlabeled_idx[new_indices]))


class CoreSetMIPSampling(QueryMethod):
    """
    An implementation of the core set query strategy with the MIP formulation using gurobi as our optimization solver.
    """

    def __init__(self, model, input_shape, num_labels, gpu):
        super().__init__(model, input_shape, num_labels, gpu)
        self.subsample = False

    def greedy_k_center(self, labeled, unlabeled, amount):

        greedy_indices = []

        # get the minimum distances between the labeled and unlabeled examples (iteratively, to avoid memory issues):
        min_dist = np.min(distance_matrix(labeled[0, :].reshape((1, labeled.shape[1])), unlabeled), axis=0)
        min_dist = min_dist.reshape((1, min_dist.shape[0]))
        for j in range(1, labeled.shape[0], 100):
            if j + 100 < labeled.shape[0]:
                dist = distance_matrix(labeled[j:j+100, :], unlabeled)
            else:
                dist = distance_matrix(labeled[j:, :], unlabeled)
            min_dist = np.vstack((min_dist, np.min(dist, axis=0).reshape((1, min_dist.shape[1]))))
            min_dist = np.min(min_dist, axis=0)
            min_dist = min_dist.reshape((1, min_dist.shape[0]))

        # iteratively insert the farthest index and recalculate the minimum distances:
        farthest = np.argmax(min_dist)
        greedy_indices.append(farthest)
        for i in range(amount-1):
            if i%1000==0:
                print("At Point " + str(i))
            dist = distance_matrix(unlabeled[greedy_indices[-1], :].reshape((1,unlabeled.shape[1])), unlabeled)
            min_dist = np.vstack((min_dist, dist.reshape((1, min_dist.shape[1]))))
            min_dist = np.min(min_dist, axis=0)
            min_dist = min_dist.reshape((1, min_dist.shape[0]))
            farthest = np.argmax(min_dist)
            greedy_indices.append(farthest)

        return np.array(greedy_indices, dtype=int), np.max(min_dist)

    def get_distance_matrix(self, X, Y):

        x_input = K.placeholder((X.shape))
        y_input = K.placeholder(Y.shape)
        dot = K.dot(x_input, K.transpose(y_input))
        x_norm = K.reshape(K.sum(K.pow(x_input, 2), axis=1), (-1, 1))
        y_norm = K.reshape(K.sum(K.pow(y_input, 2), axis=1), (1, -1))
        dist_mat = x_norm + y_norm - 2.0*dot
        sqrt_dist_mat = K.sqrt(K.clip(dist_mat, min_value=0, max_value=10000))
        dist_func = K.function([x_input, y_input], [sqrt_dist_mat])

        return dist_func([X, Y])[0]

    def get_neighborhood_graph(self, representation, delta):

        graph = {}
        print(representation.shape)
        for i in range(0, representation.shape[0], 1000):

            if i+1000 > representation.shape[0]:
                distances = self.get_distance_matrix(representation[i:], representation)
                amount = representation.shape[0] - i
            else:
                distances = self.get_distance_matrix(representation[i:i+1000], representation)
                amount = 1000

            distances = np.reshape(distances, (amount, -1))
            for j in range(i, i+amount):
                graph[j] = [(idx, distances[j-i, idx]) for idx in np.reshape(np.where(distances[j-i, :] <= delta),(-1))]

        print("Finished Building Graph!")
        return graph

    def get_graph_max(self, representation, delta):

        print("Getting Graph Maximum...")

        maximum = 0
        for i in range(0, representation.shape[0], 1000):
            print("At Point " + str(i))

            if i+1000 > representation.shape[0]:
                distances = self.get_distance_matrix(representation[i:], representation)
            else:
                distances = self.get_distance_matrix(representation[i:i+1000], representation)

            distances = np.reshape(distances, (-1))
            distances[distances > delta] = 0
            maximum = max(maximum, np.max(distances))

        return maximum

    def get_graph_min(self, representation, delta):

        print("Getting Graph Minimum...")

        minimum = 10000
        for i in range(0, representation.shape[0], 1000):
            print("At Point " + str(i))

            if i+1000 > representation.shape[0]:
                distances = self.get_distance_matrix(representation[i:], representation)
            else:
                distances = self.get_distance_matrix(representation[i:i+1000], representation)

            distances = np.reshape(distances, (-1))
            distances[distances < delta] = 10000
            minimum = min(minimum, np.min(distances))

        return minimum

    def mip_model(self, representation, labeled_idx, budget, delta, outlier_count, greedy_indices=None):

        import gurobipy as gurobi

        model = gurobi.Model("Core Set Selection")

        # set up the variables:
        points = {}
        outliers = {}
        for i in range(representation.shape[0]):
            if i in labeled_idx:
                points[i] = model.addVar(ub=1.0, lb=1.0, vtype="B", name="points_{}".format(i))
            else:
                points[i] = model.addVar(vtype="B", name="points_{}".format(i))
        for i in range(representation.shape[0]):
            outliers[i] = model.addVar(vtype="B", name="outliers_{}".format(i))
            outliers[i].start = 0

        # initialize the solution to be the greedy solution:
        if greedy_indices is not None:
            for i in greedy_indices:
                points[i].start = 1.0

        # set the outlier budget:
        model.addConstr(sum(outliers[i] for i in outliers) <= outlier_count, "budget")

        # build the graph and set the constraints:
        model.addConstr(sum(points[i] for i in range(representation.shape[0])) == budget, "budget")
        neighbors = {}
        graph = {}
        print("Updating Neighborhoods In MIP Model...")
        for i in range(0, representation.shape[0], 1000):
            print("At Point " + str(i))

            if i+1000 > representation.shape[0]:
                distances = self.get_distance_matrix(representation[i:], representation)
                amount = representation.shape[0] - i
            else:
                distances = self.get_distance_matrix(representation[i:i+1000], representation)
                amount = 1000

            distances = np.reshape(distances, (amount, -1))
            for j in range(i, i+amount):
                graph[j] = [(idx, distances[j-i, idx]) for idx in np.reshape(np.where(distances[j-i, :] <= delta),(-1))]
                neighbors[j] = [points[idx] for idx in np.reshape(np.where(distances[j-i, :] <= delta),(-1))]
                neighbors[j].append(outliers[j])
                model.addConstr(sum(neighbors[j]) >= 1, "coverage+outliers")

        model.__data = points, outliers
        model.Params.MIPFocus = 1
        model.params.TIME_LIMIT = 180

        return model, graph

    def mip_model_subsample(self, data, subsample_num, budget, dist, delta, outlier_count, greedy_indices=None):

        import gurobipy as gurobi

        model = gurobi.Model("Core Set Selection")

        # calculate neighberhoods:
        data_1, data_2 = np.where(dist <= delta)

        # set up the variables:
        points = {}
        outliers = {}
        for i in range(data.shape[0]):
            if i >= subsample_num:
                points[i] = model.addVar(ub=1.0, lb=1.0, vtype="B", name="points_{}".format(i))
            else:
                points[i] = model.addVar(vtype="B", name="points_{}".format(i))
        for i in range(data.shape[0]):
            outliers[i] = model.addVar(vtype="B", name="outliers_{}".format(i))
            outliers[i].start = 0

        # initialize the solution to be the greedy solution:
        if greedy_indices is not None:
            for i in greedy_indices:
                points[i].start = 1.0

        # set up the constraints:
        model.addConstr(sum(points[i] for i in range(data.shape[0])) == budget, "budget")
        neighbors = {}
        for i in range(data.shape[0]):
            neighbors[i] = []
            neighbors[i].append(outliers[i])
        for i in range(len(data_1)):
            neighbors[data_1[i]].append(points[data_2[i]])
        for i in range(data.shape[0]):
            model.addConstr(sum(neighbors[i]) >= 1, "coverage+outliers")
        model.addConstr(sum(outliers[i] for i in outliers) <= outlier_count, "budget")
        model.setObjective(sum(outliers[i] for i in outliers), gurobi.GRB.MINIMIZE)

        model.__data = points, outliers
        model.Params.MIPFocus = 1

        return model

    def query_regular(self, X_train, Y_train, labeled_idx, amount):

        import gurobipy as gurobi

        unlabeled_idx = get_unlabeled_idx(X_train, labeled_idx)

        # use the learned representation for the k-greedy-center algorithm:
        representation_model = Model(inputs=self.model.input, outputs=self.model.get_layer('softmax').input)
        representation = representation_model.predict(X_train, batch_size=128, verbose=0)
        print("Calculating Greedy K-Center Solution...")
        new_indices, max_delta = self.greedy_k_center(representation[labeled_idx], representation[unlabeled_idx], amount)
        new_indices = unlabeled_idx[new_indices]
        outlier_count = int(X_train.shape[0] / 10000)
        # outlier_count = 250
        submipnodes = 20000

        # iteratively solve the MIP optimization problem:
        eps = 0.01
        upper_bound = max_delta
        lower_bound = max_delta / 2.0
        print("Building MIP Model...")
        model, graph = self.mip_model(representation, labeled_idx, len(labeled_idx) + amount, upper_bound, outlier_count, greedy_indices=new_indices)
        model.Params.SubMIPNodes = submipnodes
        points, outliers = model.__data
        model.optimize()
        indices = [i for i in graph if points[i].X == 1]
        current_delta = upper_bound
        while upper_bound - lower_bound > eps:

            print("upper bound is {ub}, lower bound is {lb}".format(ub=upper_bound, lb=lower_bound))
            if model.getAttr(gurobi.GRB.Attr.Status) in [gurobi.GRB.INFEASIBLE, gurobi.GRB.TIME_LIMIT]:
                print("Optimization Failed - Infeasible!")

                lower_bound = max(current_delta, self.get_graph_min(representation, current_delta))
                current_delta = (upper_bound + lower_bound) / 2.0

                del model
                gc.collect()
                model, graph = self.mip_model(representation, labeled_idx, len(labeled_idx) + amount, current_delta, outlier_count, greedy_indices=indices)
                points, outliers = model.__data
                model.Params.SubMIPNodes = submipnodes

            else:
                print("Optimization Succeeded!")
                upper_bound = min(current_delta, self.get_graph_max(representation, current_delta))
                current_delta = (upper_bound + lower_bound) / 2.0
                indices = [i for i in graph if points[i].X == 1]

                del model
                gc.collect()
                model, graph = self.mip_model(representation, labeled_idx, len(labeled_idx) + amount, current_delta, outlier_count, greedy_indices=indices)
                points, outliers = model.__data
                model.Params.SubMIPNodes = submipnodes

            if upper_bound - lower_bound > eps:
                model.optimize()

        return np.array(indices)

    def query_subsample(self, X_train, Y_train, labeled_idx, amount):

        import gurobipy as gurobi

        unlabeled_idx = get_unlabeled_idx(X_train, labeled_idx)

        submipnodes = 20000
        subsample_num = 30000
        subsample_idx = np.random.choice(unlabeled_idx, subsample_num, replace=False)
        subsample = np.vstack((X_train[labeled_idx], X_train[subsample_idx]))
        new_labeled_idx = np.arange(len(labeled_idx))
        new_indices = self.query_regular(subsample, Y_train, new_labeled_idx, amount)
        return np.array(subsample_idx[new_indices - len(labeled_idx)])


    def query(self, X_train, Y_train, labeled_idx, amount):

        if self.subsample:
            return self.query_subsample(X_train, Y_train, labeled_idx, amount)
        else:
            return self.query_regular(X_train, Y_train, labeled_idx, amount)


class EGLSampling(QueryMethod):
    """
    An implementation of the EGL query strategy.
    """

    def __init__(self, model, input_shape, num_labels, gpu):
        super().__init__(model, input_shape, num_labels, gpu)

    def compute_egls(self, unlabeled, n_classes):

        # create a function for computing the gradient length:
        self.input_placeholder = K.placeholder(self.model.get_input_shape_at(0))
        self.output_placeholder = K.placeholder(self.model.get_output_shape_at(0))
        predict = self.model.call(self.input_placeholder)
        loss = K.mean(categorical_crossentropy(self.output_placeholder, predict))
        weights = [tensor for tensor in self.model.trainable_weights]
        gradient = self.model.optimizer.get_gradients(loss, weights)
        gradient_flat = [K.flatten(x) for x in gradient]
        gradient_flat = K.concatenate(gradient_flat)
        gradient_length = K.sum(K.square(gradient_flat))
        self.get_gradient_length = K.function([K.learning_phase(), self.input_placeholder, self.output_placeholder], [gradient_length])

        # calculate the expected gradient length of the unlabeled set (iteratively, to avoid memory issues):
        unlabeled_predictions = self.model.predict(unlabeled)
        egls = np.zeros(unlabeled.shape[0])
        for i in range(n_classes):
            calculated_so_far = 0
            while calculated_so_far < unlabeled_predictions.shape[0]:
                if calculated_so_far + 100 >= unlabeled_predictions.shape[0]:
                    next = unlabeled_predictions.shape[0] - calculated_so_far
                else:
                    next = 100

                labels = np.zeros((next, n_classes))
                labels[:,i] = 1
                grads = self.get_gradient_length([0, unlabeled[calculated_so_far:calculated_so_far+next, :], labels])[0]
                grads *= unlabeled_predictions[calculated_so_far:calculated_so_far+next, i]
                egls[calculated_so_far:calculated_so_far+next] += grads

                calculated_so_far += next

        return egls

    def query(self, X_train, Y_train, labeled_idx, amount):

        unlabeled_idx = get_unlabeled_idx(X_train, labeled_idx)
        n_classes = Y_train.shape[1]

        # choose the samples with the highest expected gradient length:
        egls = self.compute_egls(X_train[unlabeled_idx], n_classes)
        selected_indices = np.argpartition(egls, -amount)[-amount:]
        return np.hstack((labeled_idx, unlabeled_idx[selected_indices]))


class CombinedSampling(QueryMethod):
    """
    An implementation of a query strategy which naively combines two given query strategies, sampling half of the batch
    from one strategy and the other half from the other strategy.
    """

    def __init__(self, model, input_shape, num_labels, method1, method2, gpu):
        super().__init__(model, input_shape, num_labels, gpu)
        self.method1 = method1(model, input_shape, num_labels, gpu)
        self.method2 = method2(model, input_shape, num_labels, gpu)

    def query(self, X_train, Y_train, labeled_idx, amount):
        labeled_idx = self.method1.query(X_train, Y_train, labeled_idx, int(amount/2))
        return self.method2.query(X_train, Y_train, labeled_idx, int(amount/2))

    def update_model(self, new_model):
        del self.model
        gc.collect()
        self.model = new_model
        self.method1.update_model(new_model)
        self.method2.update_model(new_model)


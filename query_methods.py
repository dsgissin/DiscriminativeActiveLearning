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

    def __init__(self, model, input_shape=(28,28), num_labels=10):
        self.model = model
        self.input_shape = input_shape
        self.num_labels = num_labels

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
    A basic random sampling query strategy baseline.
    """

    def __init__(self, model, input_shape, num_labels):
        super().__init__(model, input_shape, num_labels)

    def query(self, X_train, Y_train, labeled_idx, amount):
        unlabeled_idx = get_unlabeled_idx(X_train, labeled_idx)
        return np.hstack((labeled_idx, np.random.choice(unlabeled_idx, amount, replace=False)))


class UncertaintySampling(QueryMethod):
    """
    The basic uncertainty sampling query strategy, querying the examples with the minimal top confidence.
    """

    def __init__(self, model, input_shape, num_labels):
        super().__init__(model, input_shape, num_labels)

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

    def __init__(self, model, input_shape, num_labels):
        super().__init__(model, input_shape, num_labels)

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

    def __init__(self, model, input_shape, num_labels):
        super().__init__(model, input_shape, num_labels)

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

        unlabeled_predictions = np.amax(predictions, axis=1)

        selected_indices = np.argpartition(unlabeled_predictions, amount)[:amount]
        return np.hstack((labeled_idx, unlabeled_idx[selected_indices]))


class BayesianUncertaintyEntropySampling(QueryMethod):
    """
    An implementation of the Bayesian active learning method, using maximal entropy as the decision rule.
    """

    def __init__(self, model, input_shape, num_labels):
        super().__init__(model, input_shape, num_labels)

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

    def __init__(self, model, input_shape, num_labels):
        super().__init__(model, input_shape, num_labels)

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

    def __init__(self, model, input_shape, num_labels):
        super().__init__(model, input_shape, num_labels)

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

            model = train_discriminative_model(X_train[labeled_idx], X_train[unlabeled_idx], self.input_shape)
            predictions = model.predict(X_train[unlabeled_idx])
            selected_indices = np.argpartition(predictions[:,1], -sub_sample_size)[-sub_sample_size:]
            labeled_idx = np.hstack((labeled_idx, unlabeled_idx[selected_indices]))
            labeled_so_far += sub_sample_size

            # delete the model to free GPU memory:
            del model
            gc.collect()

        return labeled_idx


class DiscriminativeRepresentationSampling(QueryMethod):
    """
    An implementation of DAL (discriminative active learning), using the learned representation as our representation.
    This implementation is the one which performs best in practice.
    """

    def __init__(self, model, input_shape, num_labels):
        super().__init__(model, input_shape, num_labels)

        self.sub_batches = 25


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

            model = train_discriminative_model(representation[labeled_idx], representation[unlabeled_idx], representation[0].shape)
            predictions = model.predict(representation[unlabeled_idx])
            selected_indices = np.argpartition(predictions[:,1], -sub_sample_size)[-sub_sample_size:]
            labeled_idx = np.hstack((labeled_idx, unlabeled_idx[selected_indices]))
            labeled_so_far += sub_sample_size

            # delete the model to free GPU memory:
            del model
            gc.collect()
        del embedding_model

        return labeled_idx


class DiscriminativeAutoencoderSampling(QueryMethod):
    """
    An implementation of DAL (discriminative active learning), using an autoencoder embedding as our representation.
    """

    def __init__(self, model, input_shape, num_labels):
        super().__init__(model, input_shape, num_labels)

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

            model = train_discriminative_model(self.embedding[labeled_idx], self.embedding[unlabeled_idx], self.embedding[0].shape)
            predictions = model.predict(self.embedding[unlabeled_idx])
            selected_indices = np.argpartition(predictions[:,1], -sub_sample_size)[-sub_sample_size:]
            labeled_idx = np.hstack((labeled_idx, unlabeled_idx[selected_indices]))
            labeled_so_far += sub_sample_size

            # delete the model to free GPU memory:
            del model
            gc.collect()

        return labeled_idx


class CoreSetSampling(QueryMethod):
    """
    An implementation of the greedy core set query strategy.
    """

    def __init__(self, model, input_shape, num_labels):
        super().__init__(model, input_shape, num_labels)

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

    def __init__(self, model, input_shape, num_labels):
        super().__init__(model, input_shape, num_labels)

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

        return np.array(greedy_indices, dtype=int), np.max(min_dist)


    def mip_model(self, data, subsample_num, budget, dist, delta, outlier_count, greedy_indices=None):

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

        return model


    def query(self, X_train, Y_train, labeled_idx, amount):

        import gurobipy as gurobi

        unlabeled_idx = get_unlabeled_idx(X_train, labeled_idx)

        submipnodes = 20000
        subsample_num = 3000
        subsample_idx = np.random.choice(unlabeled_idx, subsample_num, replace=False)
        subsample = np.vstack((X_train[subsample_idx], X_train[labeled_idx]))

        # use the learned representation for the k-greedy-center algorithm:
        representation_model = Model(inputs=self.model.input, outputs=self.model.get_layer('softmax').input)
        representation = representation_model.predict(subsample, batch_size=256, verbose=0)
        new_indices, max_delta = self.greedy_k_center(representation[subsample_num:, :], representation[:subsample_num, :], amount)
        dist = distance_matrix(representation, representation)
        # outlier_count = int(representation.shape[0] / 100)
        outlier_count = 250

        # iteratively solve the MIP optimization problem:
        eps = 0.000001
        upper_bound = max_delta
        lower_bound = max_delta / 2.0
        model = self.mip_model(representation, subsample_num, len(labeled_idx) + amount, dist, upper_bound, outlier_count, greedy_indices=new_indices)
        model.Params.SubMIPNodes = submipnodes
        points, outliers = model.__data
        model.optimize()
        indices = [i for i in range(subsample.shape[0]) if points[i].X == 1]
        while upper_bound - lower_bound > eps:
            current_delta = (upper_bound + lower_bound) / 2.0

            print("upper bound is {ub}, lower bound is {lb}".format(ub=upper_bound, lb=lower_bound))
            model.optimize()
            if model.getAttr(gurobi.GRB.Attr.Status) == gurobi.GRB.INFEASIBLE:
                print("Optimization Failed - Infeasible!")
                lower_bound = max(current_delta, np.min(dist[dist>=current_delta]))
                current_delta = (upper_bound + lower_bound) / 2.0
                model = self.mip_model(representation, subsample_num, len(labeled_idx) + amount, dist, current_delta, outlier_count, greedy_indices=indices)
                model.Params.SubMIPNodes = submipnodes

            else:
                print("Optimization Succeeded!")
                upper_bound = min(current_delta, np.max(dist[dist<=current_delta]))
                current_delta = (upper_bound + lower_bound) / 2.0
                points, outliers = model.__data
                indices = [i for i in range(subsample.shape[0]) if points[i].X == 1]
                model = self.mip_model(representation, subsample_num, len(labeled_idx) + amount, dist, current_delta, outlier_count, greedy_indices=indices)
                model.Params.SubMIPNodes = submipnodes

        indices = np.array(indices)
        return np.hstack((labeled_idx, np.array(subsample_idx[indices[indices < subsample_num]])))


class EGLSampling(QueryMethod):

    def __init__(self, model, input_shape, num_labels):
        super().__init__(model, input_shape, num_labels)


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

    def __init__(self, model, input_shape, num_labels, method1, method2):
        super().__init__(model, input_shape, num_labels)
        self.method1 = method1(model, input_shape, num_labels)
        self.method2 = method2(model, input_shape, num_labels)

    def query(self, X_train, Y_train, labeled_idx, amount):
        labeled_idx = self.method1.query(X_train, Y_train, labeled_idx, int(amount/2))
        return self.method2.query(X_train, Y_train, labeled_idx, int(amount/2))

    def update_model(self, new_model):
        del self.model
        gc.collect()
        self.model = new_model
        self.method1.update_model(new_model)
        self.method2.update_model(new_model)


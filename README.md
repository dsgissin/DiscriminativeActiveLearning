# Discriminative Active Learning

This repository contains the code used to run the deep active learning experiments detailed in our [blog](https://dsgissin.github.io/DiscriminativeActiveLearning/).

You may use the code in this repository, but note that this isn't a complete active learning library and is not fully generic. Replicating the experiments and using the implementations should be easy, but adapting the code to new datasets and experiment types may take a bit of effort.

## Dependencies

In order to run our code, you'll need these main packages:

- [Python](https://www.python.org/)>=3.5
- [Numpy](http://www.numpy.org/)>=1.14.3
- [Scipy](https://www.scipy.org/)>=1.0.0
- [TensorFlow](https://www.tensorflow.org/)>=1.5
- [Keras](https://keras.io/)>=2.2
- [Gurobi](http://www.gurobi.com/documentation/)>=8.0 (for the core set MIP query strategy)
- [Cleverhans](https://github.com/tensorflow/cleverhans)>=2.1 (for the adversarial query strategy

## Running the Code

The code is run using the main.py file in the following way:

    python3 main.py <experiment_index> <dataset> <batch_size> <initial_size> <iterations> <method> <experiment_folder> -method2 <method2> -idx "/cs/labs/shais/dsgissin/ActiveLearning/experiment_indices/" -gpu <gpus>

- experiment_index: an integer detailing the number of experiment (since usually many are run in parallel and combined later).
- dataset: a string detailing the dataset for this experiment (one of "mnist", "cifar10" or "cifar100").
- batch_size: the size of the batch of examples to be labeled in every iteration.
- initial_size: the amount of labeled examples to start the experiment with (chosen randomly).
- iteration: the amount of active learning iterations to run in the experiment.
- method: a string for the name of the query strategy to be used in the experiment.
- experiment_folder: the path of the folder where the experiment data and results will be saved.

There are also three optional parameters:
- idx: a path to the folder with the pickle file containing the initial labeled example indices for the experiment.
- method2: the name of the second query strategy (if you want to try and combine two methods together).
- gpu: the number of gpus to use for training the models.

### Possible Method Names
These are the possible names of methods that can be used in the experiments:
- "Random": random sampling
- "CoreSet": the greedy core set approach
- "CoreSetMIP": the core set with the MIP formulation
- "Discriminative": discriminative active learning with raw pixels as the representation
- "DiscriminativeAE": discriminative active learning with an autoencoder embedding as the representation
- "DiscriminativeLearned": discriminative active learning with the learned representation from the model as the representation
- "DiscriminativeStochastic": discriminative active learning with the learned representation as the representation and sampling proportionally to the confidence as being "unlabeled".
- "Uncertainty": uncertainty sampling with minimal top confidence
- "UncertaintyEntropy": uncertainty sampling with maximal entropy
- "Bayesian": Bayesian uncertainty sampling with minimal top confidence
- "BayesianEntropy": Bayesian uncertainty sampling with maximal entropy
- "EGL": estimated gradient length
- "Adversarial": adversarial active learning using DeepFool


## Directory Structure

### main.py

This file contains the logic which runs the active learning experiment and saves the results to the relevant folder.

### models.py

This file contains all of the neural network models and training functions used by the query methods.

### query_methods.py

this file contains the query strategy implementations for all of the methods detailed in the blog.

## Examples


    python3 main.py 0 "mnist" 100 100 20 "Random" "/path/to/experiment/folder" -idx "/path/to/folder/with/initial/index/file"
    python3 main.py 7 "cifar10" 5000 5000 5 "DiscriminativeLearned" "/path/to/experiment/folder" -idx "/path/to/folder/with/initial/index/file"
    python3 main.py 0 "cifar100" 5000 5000 3 "Adversarial" "/path/to/experiment/folder" -method2 "Bayesian" -gpu 2


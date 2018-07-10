---
layout: post
title: Comparing The Methods
---
## Disclaimers
In the [previous post][previous post] we surveyed some of today's state of the art active learning methods, applied to deep learning models. In this post we will compare the methods in different situations and try to understand which is better. If you're interested in our implementation of these methods and in code that can recreate these experiments, you're welcome to check out the [github repository][git] which contains all of the code.

Before we get into the experimental setup and the results, we should make a few disclaimers about the experiments and active learning experiments in general. When we run active learning experiments using neural networks, there are many random things that effect a single experiment. First, the initial random batch strongly effects the first batch query (and subsequent batches as well). Next we have that the neural network which trains on the labeled set is a non-convex model, and so different random seeds will lead to different models (and embeddings). This in turn causes another problem - when we want to evaluate the test accuracy of a labeled set, different evaluations (which are training a neural net on the labeled set) will lead to different accuracies...

All of these random effects and others mean that we should be extra careful when running these kinds of experiments. Sadly, most papers on this subject do not specify their complete experimental setup, leaving many things open to interpretation. Usually we are lucky to get more than the model and basic hyper parameters that were used, and this meant we had to try and devise a setup which is as fair as possible to the different methods.

Another important parameter for the experiments that changes between the papers is the **batch size**. As we saw in our [last post][previous post], having a large batch means that there is a bigger chance of the chosen examples being correlated and so the batch becomes less effective at raising our accuracy. Not surprisingly, papers that propose greedy active learning methods use a relatively small batch size while papers that propose batch-aware methods use a very large batch size. These choices help in showing the methods' stronger sides, but they make comparing the methods harder. We tried to compare the methods in both types of batch sizes in as fair a way as possible.

As you will see, **our experimental results are different from some of the results reported in the papers**. When these differences occur we will point them out and try to give an explanation, although in some of the cases we won't have a good one.

## Experimental Setup
So after that long disclaimer, how did we choose to setup the experiments?

A single experiment for a method runs in the following way: 

- The method starts with a random initial labeled set and a large unlabeled set.

- A neural network is trained on the labeled set and the resulting model is handed over to the method. The model architecture and hyper parameters are the same for all of the models.

- The model's accuracy on a held out test set is recorded. This test set is the same for all of the experiments and methods.

- The method uses the model to choose a batch to be labeled and that batch is added to the labeled set.

- Repeat from step 2 for a predetermined amount of iterations.

We run ~10 experiments for every one of the methods and then average their accuracy in every iteration over all of the experiments. The plots in this post present the average accuracy for every iteration, along with their empirical standard deviations.

We'll now move to some final nitty details of an experiment.

#### Initial Labeled Set
The initial labeled set has a big effect on the entire experiment. To make things fair, we used the same initial labeled set for all of the methods. For example, if we ran 10 experiments for all of the methods, there would be 10 initial labeled sets that would be the same for all of the methods. This insures that we control for the effects of the randomness of the initial labeled set.

#### Training a Neural Network on the Labeled Set
As we said, the neural network training can have different results on the same labeled set. We tried experimenting with many forms of early stopping and running several evaluations and taking the best one, but these didn't seem to work very well (mostly due to the difficulty of stopping the training at the right time).

The method we found that works best was to train a neural network only once, but to train it for much longer than necessary while saving the weights during training after every epoch where the validation accuracy has improved (or stayed the same). The validation set was always a random subset of 20% of the labeled set. This allowed us to not worry about early stopping issues and made sure that we got the best (or close to the best) weights of the model.

This does not completely control the randomness of the neural network training, but we found it reduces it quite a bit and we are able to deal with the randomness by running several experiments and averaging over them...


## Comparison on MNIST
We start with a comparison on the MNIST dataset. The architecture we use here for all of the methods is a simple convolutional architecture (the LeNet architecture). For a detailing of the hyper parameters you're welcome to look at [our code][git]. The batch size we used here was a **batch size of 100**, and an initial labeled set of size 100 as well.

The methods we compared are random sampling, regular uncertainty sampling (using the softmax scores), Bayesian uncertainty sampling, Adversarial active learning, EGL (as detailed shortly in the [first post][first post]), core set with the greedy algorithm and core set with the MIP extension.

First, let's compare the methods separately against random sampling, the usual baseline.

### Uncertainty Sampling Methods
We'd like to see how much does the Bayesian approach to uncertainty estimation in neural networks improves performance in active learning in comparison to the regular uncertainty sampling which simply uses the softmax scores. In the following plot we compare the different methods, along with the different possible decision rules (top confidence and max entropy):

TODO: add uncertainty comparison
TODO: compare to entropy as well?
TODO: is this different than results in the papers?

### Adversarial Active Learning
We have only one formulation of this method that we checked, which corresponds to the formulation in the paper. Let's see how it compares to random sampling:

TODO: compare to random sampling...

### Core Set Methods
Next, we compare the greedy core set method with the MIP formulation. As detailed in the core set paper, adding the MIP formulation increases the running time of the core set algorithm considerably (and we saw this in our experiments as well). Also, during these experiments we were unable to run the MIP formulation on the full MNIST dataset (which has 50,000 examples), due to memory issues. This can be resolved by using frameworks that handle large optimization problems such as this, but this was outside the scope of this project. We chose to **subsample 3000 unlabeled examples** and run the MIP formulation on that subset (along with the labeled set). This is different than what was reported in the core set paper, and so our results might be different due to this difference. We also changed the amount of outliers allowed in the formulation to be a constant of **250 outliers**, which is also different from what was reported in the paper.

After those disclaimers we can move on to comparing the two formulations:

{% include image.html path="Experiments/results_mnist_CoreSet.png" %}

Well, we can see that the MIP formulation has a very positive effect from very early on in the active learning process, which is very nice to see. This improvement is consistant with what was reported in the core set paper, and they were able to use the entire dataset and not a subsample like us. We should note that while we haven't explored this idea enough to make any definitive claims, we feel that this improvement is brought on mainly due to the robustness to outliers of the MIP formulation.

So we can conclude that if you are OK with waiting a bit before getting the next batch of examples to be labeled, and if you have the resources to solve a MIP optimization problem with many variables, it is worth it to use the MIP formulation. Still, it's nice to see that the method beats random sampling even in it's greedy formulation, although the difference isn't very large. This is mostly due to the relatively small batch size, which means the method's batch-aware nature is less present.

### EGL
We implemented EGL and compared it to random sampling:

{% include image.html path="Experiments/results_mnist_EGL.png" %}

Oh. This looks rather bad. This was also the case in the adversarial paper when they tried to compare their method to EGL, but the EGL paper shows great improvements against both random sampling and other methods, so what happened?

We'll explain what we think happened later in the post.

### Full Comparison
TODO: what's different than reported in the papers?

## CIFAR-10 Comparison
which model was used? batch size? initial set size? why no MIP comparison?

## CIFAR-100 Comparison
which model was used?

## Batch Composition & Label Entropy
TODO: EGL was designed for speech recognition which uses a MLE loss function and an RNN, which is a very different formulation than what we had here... There are also many more labels (they margenalize over the top 100)

## Summary


[previous post]: https://dsgissin.github.io/DiscriminativeActiveLearning/2018/07/05/Batch-AL.html
[git]: https://github.com/dsgissin/DiscriminativeActiveLearning
[first post]: https://dsgissin.github.io/DiscriminativeActiveLearning/2018/07/05/AL-Intro.html
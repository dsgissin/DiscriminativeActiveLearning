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
We start with a comparison on the [MNIST dataset][MNIST]. The architecture we use here for all of the methods is a simple convolutional architecture (the LeNet architecture). For a detailing of the hyper parameters you're welcome to look at [our code][git]. The batch size we used here was a **batch size of 100**, and an initial labeled set of size 100 as well.

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

Well, we can see that the MIP formulation has a very positive effect from very early on in the active learning process, which is very nice to see. This improvement is consistant with what was reported in the core set paper, even though we weren't able to use the entire unlabeled set like they did. We should note that while we haven't explored this idea enough to make any definitive claims, we feel that this improvement is brought on mainly due to the robustness to outliers of the MIP formulation.

So we can conclude that if you are OK with waiting a bit before getting the next batch of examples to be labeled, and if you have the resources to solve a MIP optimization problem with many variables, it is worth it to use the MIP formulation. Still, it's nice to see that the method beats random sampling even in it's greedy formulation, although the difference isn't very large. This is mostly due to the relatively small batch size, which means the method's batch-aware nature is less present.

### EGL
We implemented EGL and compared it to random sampling:

{% include image.html path="Experiments/results_mnist_EGL.png" %}

Oh. This looks bad. 

This was also the case in the adversarial paper when they tried to compare their method to EGL, but the EGL paper shows great improvements against both random sampling and other methods, so what happened?

We'll explain what we think happened later in the post.

### Full Comparison
TODO: what's different than reported in the papers? why is uncertainty so good?

But all of this is just a comparison on MNIST, the most worn out dataset in history. We'd like to compare the methods on more realistic datasets and with a larger batch size, which simulates real-life active learning problems better. For that we turn to a image classification dataset that is only a bit less worn out - [CIFAR][CIFAR]!

## CIFAR-10 Comparison
CIFAR-10 is a image classification dataset with 10 classes. As opposed to the MNIST dataset, the images here are RGB and a bit larger (32x32). This task is harder and so we need a better, deeper architecture to solve it. We chose to follow the lead of the core set paper and use the **VGG-16 architecture**, a deep convolutional neural network which is rather popular in general as a pre-trained feature extractor. We used a much larger batch size this time - a **batch size of 5000**, along with an initial sample of 5000 examples.

We chose this batch size both because it was the setting in the core set paper, and because it gives us a view of a batch size that is orders of magnitude larger that the one usually used when comparing these methods. We also believe that this batch size is much more realistic for industry applications, where the datsets are usually very big and the labeling is paralleled.

However because the batch size is so big here, our computational constraints made it impossible to use the MIP formulation of the core set approach (5000 examples was the largest amount we were able to work with on the MNIST experiments). While we hoped to compare the methods in a fair way, we take comfort in the fact that this is the exact setting used in the core set paper and the results we got in other experiments are similar to those in the paper, so we refer to the author's results. They report that the MIP formulation improves the greedy results during the experiments by ~2% throughout the active learning process.



## CIFAR-100 Comparison
which model was used?

## Batch Composition & Label Entropy
In this section we'll try to learn more about the behavior of the algorithms by looking at how a queried batch is distributed across the different classes. We did this both as a sanity measure and to try and understand why EGL's results were so bad, under the hypothesis that it is because there is very high correlation between the examples that are queried by EGL.

To summarize the distribution of labels in the examples that were queried with a single scalar, we'll define the notion of "label entropy". The label entropy of a set of \\(M\\) examples will be the entropy of the empirical distribution of labels in the set. Mathematically, this can be written as:

$$ LabelEntropy(S) = - \sum_{y \in Y}\frac{Count(y)}{M}log(\frac{Count(y)}{M}) $$

Like in regular entropy, a high label entropy corresponds to higher uncertainty on the labels in the set. Since we want active learning methods to create a labeled set that is relatively diverse and represents the full distribution well, we should expect a good active learning method to create a labeled dataset which has a similar label distribution as the full dataset. Since our full dataset has the same amount of images for every label (maximal entropy), we would say that as a rule of thumb, higher label entropy during the active learning process is better.

It isn't very difficult to record this label entropy in our experiments, and so we recorded the entropy of the different algorithms. In every iteration of the experiment we calculated the label entropy of all of the examples queried so far, and in the following plot we can see the results averaged over several experiments (the dotted black line is the maximal entropy over the number of classes):

{% include image.html path="Experiments/results_mnist_entropy_cumulative.png" %}

TODO: insert picture of CIFAR here

This is very interesting - except for the random sampling which has the highest label entropy (unsurprisingly), we see a **one to one correlation between the accuracy ranking of the methods and their label entropy!** This is quite pleasing and shows that this measure can be used as a proxy for the success of an active learning method practice. If the method you're using has a label distribution that is far from the label distribution you expect for your data, you should suspect that the method isn't working very well...

Another interesting thing to note here is the label entropy of EGL which is much, much lower than all of the others, and barely improves during the process. This confirms our suspicions - EGL continuously queries examples from the same few classes, and does not lead to a balanced labeled dataset. This explains the poor performance it had during our experiments...

But the EGL paper showed great results, so what happened?

Remember that the team at Baidu implemented EGL as an active learning method for speech recognition using RNNs, and our experiments were all on image classification tasks using CNNs. The two architectures are very different and so it is quite plausible that the architectures have gradients that behave differently. There is also a difference in the loss function used, where Baidu's team tries to maximize the log likelihood of the data while we try to minimize the cross entropy classification loss. With all of these differences, it isn't surprising that the gradients will behave differently between the experiments, and because EGL is based on the gradient length, it isn't surprising that we will see this difference in performace.

Checking that our suspicions are correct regarding the difference in gradient behavior is interesting, but is outside the scope of this project...

## Ranking Comparison
Another interesting way we can compare methods is used in the EGL paper, and we will use it here too. We can look at the way each method ranks examples in the unlabeled set from high score to low score (applicable only to the greedy methods), and try to look at the rankings. Similar methods should rank the unlabeled set in a similar way, so that if we plot the rankings one against the other we should expect them to more or less line up on the diagonal of the plot. On the other hand, methods which are very different should have plots that are all over the place.

We'll start by comparing two methods which we should expect to be similar - uncertainty sampling and Bayesian uncertainty sampling. Note that we've ordered the ranking such that those who are ranked first (bottom and left) are those that will be selected first (ranked most informative under the query strategy):

{% include image.html path="Experiments/results_mnist_ranking_uncertainty_bayesian.png" %}

Indeed, we got that the two methods rank the unlabeled examples in a very similar way. Next we can have a look at the adversarial method. It's intuition is similar to that of uncertainty sampling and as we saw in earlier posts, they are the same thing in linear models. Do they behave similarly when applied to highly non linear functions like neural nets?

{% include image.html path="Experiments/results_mnist_ranking_uncertainty_adversarial.png" %}

It seems so! 

While the two methods are very different in what they do, the eventual results are very similar. A comparison of the adversarial approach to Bayesian uncertainty sampling looks basically the same.

Finally, we'll have a look at EGL compared to uncertainty sampling. We expect there to be little correlation since we saw EGL selects examples predominately from the same class...

{% include image.html path="Experiments/results_mnist_ranking_uncertainty_egl.png" %}


## Summary
In this post we empirically compared the different methods that we detailed in the last post, using the MNIST and CIFAR datasets. We saw the effect of the batch size on the different methods, and reviewed two additional interesting ways to compare active learning methods - the label entropy and the ranking comparison.

We also saw that the methods can be sensitive to the actual objective we are optimizing, and the network architecture. This is especially true for methods like EGL, which use the model's gradient as part of their decision rule (and it might also be true of the adversarial approach for this same reason).

Overall, while the different methods perform differently in different situations, it doesn't seem like any of them really outshines the good old uncertainty sampling. This comes in contrast to what is shown in the different papers, and we do not have a good explanation for this.

In the [next post][next post], we will review the work we did on developing a new active learning method - "Discriminative Active Learning". The method didn't pan out and we weren't able to get it to work as well as we'd hoped, but the thought process could be of interest. If you would rather skip ahead, [the post after that][last post] concludes this review of active learning.


[previous post]: https://dsgissin.github.io/DiscriminativeActiveLearning/2018/07/05/Batch-AL.html
[git]: https://github.com/dsgissin/DiscriminativeActiveLearning
[first post]: https://dsgissin.github.io/DiscriminativeActiveLearning/2018/07/05/AL-Intro.html
[MNIST]: http://yann.lecun.com/exdb/mnist/
[CIFAR]: https://www.cs.toronto.edu/~kriz/cifar.html
[next post]: 
[last post]: 
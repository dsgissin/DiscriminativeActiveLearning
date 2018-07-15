---
layout: post
title: References
---

### Active Learning Surveys & Benchmarks
- [Burr Settles' Survey][survey] gives a great introduction to the classical active learning framework.

- This [active learning benchmark][logistic regression] gives a very thorough comparison of many classic active learning methods using the logistic regression classifier on many datasets. They too find that uncertainty sampling is surprisingly effective across a wide range of datasets.

### Methods We've Covered
- [Bayesian Unceratainty Sampling][bayesian]

- The Core Set Approach - [Silvio Savarese's paper][Silvio Core Set] and [Ran El-Yaniv's paper][Ran Core Set].

- [The Adversarial Approach][adversarial]

- [The EGL paper][egl] by the Baidu team

- [Coordinate Matching][coordinate]

### Other Interesting Methods

#### Ensemble Methods
As we already hinted at, due to the fact that different active learning methods sometimes put emphasis on different examples, it may be beneficial to want to combine these methods together in some way. This hasn't been attempted in a successful way in the batch framework, but there were a few papers that attempted this in the classic framework by formulating the ensemble as different arms in a [multi armed bandit][multi armed] scenario, which is a field that has been studied for many years.

If you're interested, [this paper][bandit1] and [this paper][bandit2] are good places to start.

#### Meta Learning
The field of meta learning is picking up a lot of speed in recent years, with the promise of being able to stop using algorithms based on heuristics in many fields, and replacing them with algorithms that are learned. In the end of the day, **an algorithm is a function** which takes in examples and outputs a function. Becuase it's a function, we can approximate it using soome neural network architecture and try to learn the weights that will give us a great algorithm.

Meta learning has been used to try and tackle many types of problems, and active learning is one of them. If you're interested, take a look at [this paper][meta1], [this paper][meta2] and [this paper][meta3].

#### Membership Query Synthesis Using GANs
We introduced membership query synthesis as a different framework for active learning, where the learner synthesizes new examples to be labeled by the annotator. Due to the fact that the learner needs to have a strong model of the data distribution in order to synthesize examples which can be labeled by an annotator (and which aren't pure noise), this field slowly faded away.

Thanks to huge improvements in the field of generative models brought by [GANs][gan], this is no longer such a problem and the field of membership query synthesis has been revisited with modern approaches. The results aren't mind blowing yet, but there is reason to be hopeful. You're welcome to [read the paper][gan paper].

### Applications For Optimizing Neural Networks
Finally, there is a very interesting application of active learning in the rather surprising field of optimization. Neural networks are almost exclusively trained using mini batch SGD, where the mini batch is uniformly drawn from the training set.

By now you can probably guess how this process can be improved, since random sampling is something we've spent the whole blog trying to beat. Instead of trying to get a diverse and representative labeled sample from our data distribution, we can use active learning here to get a mini batch that is most informative for training the network and will lead to faster convergence of the network (and maybe even to a better optimum).

If you want to see many of the concepts we've covered applied to active mini batch selection, you can read [this paper][batch1], [this paper][batch2], [this paper][batch3] and [this paper][batch4] (among others).



[survey]: http://burrsettles.com/pub/settles.activelearning.pdf
[logistic regression]: http://arxiv-export-lb.library.cornell.edu/pdf/1611.08618

[bayesian]: https://arxiv.org/pdf/1703.02910.pdf
[adversarial]: https://arxiv.org/pdf/1802.09841.pdf
[Ran Core Set]: https://arxiv.org/pdf/1711.00941.pdf
[Silvio Core Set]: https://arxiv.org/pdf/1708.00489.pdf
[egl]: https://arxiv.org/pdf/1612.03226.pdf
[coordinate]: https://icml.cc/2012/papers/607.pdf

[multi armed]: https://en.wikipedia.org/wiki/Multi-armed_bandit
[bandit1]: https://www.csie.ntu.edu.tw/~htlin/paper/doc/aaai15albl.pdf
[bandit2]: https://pdfs.semanticscholar.org/f761/2bb8aba0d7c6c3c90bb82da0d9df60768217.pdf

[meta1]: https://arxiv.org/pdf/1708.02383.pdf
[meta2]: https://arxiv.org/pdf/1706.08334.pdf
[meta3]: https://openreview.net/pdf?id=HJ4IhxZAb

[gan]: https://en.wikipedia.org/wiki/Generative_adversarial_network
[gan paper]: https://arxiv.org/pdf/1702.07956.pdf

[batch1]: https://arxiv.org/pdf/1511.06343.pdf
[batch2]: https://arxiv.org/pdf/1604.03540.pdf
[batch3]: https://arxiv.org/pdf/1704.07433.pdf
[batch4]: https://arxiv.org/pdf/1804.02772.pdf

---
layout: post
title: Conclusion
comments: true
---

In this blog series we've covered the framework of active learning from its initial formulation and algorithms, all the way to todays state of the art methods. We tried to be concise but there was a lot to take in, so we'll detail the most important take home messages from all of this in our mind.

### The Data & Model Matter
We saw in trying to figure out what went wrong with EGL that it performed poorly in our experiments, while it performed really well for the Baidu team on a speech recognition task. We conjectured that this was because of how the gradients behave differently for different models and machine learning tasks, and that EGL (and maybe the adversarial approach) could behave very differently on different tasks.

The thing is, **almost every research paper on modern active learning uses the same task, datasets and models** (image classification using some classic architecture). This isn't too surprising, since these datasets take a reasonable amount to time to train on and there is a lot of relevant open source code. But most people who would like to try active learning on an industry application won't use these datasets and probably won't tackle an image classification problem, and this means that **all the comparisons from the literature could be irrelevant** for their scenario, especially with the differences between the methods being relatively small.

This means that if you want to try active learning in a new domain, choosing the state of the art method on MNIST isn't necessarily the right thing to do. If you have a good amount of labeled data to start with, it could be worth the time to **experiment with the different methods** like we did here. Otherwise, you should probably **avoid the methods which are more prone to change with the domain** (EGL in particular).

### The Batch Size Doesn't Matter That Much
Contrary to what we would expect and what is presented in many papers on the subject, we didn't find that the size of the batch made a big difference on the success of the greedy algorithm. Even in very large batch sizes which are more suitable for industry applications we get great results for greedy methods such as uncertainty sampling and the adversarial approach. Increasing the batch size only helped the batch aware methods we examined (core set and ours).

While these experiments aren't enough to make a definitive claim about this, since we only tried a batch size of 5000 on CIFAR-10, they do suggest that the batch size might not be as big a problem as initially thought.

### Nothing Really Beats Uncertainty Sampling
Even though today's state of the art methods are very interesting and original, in the end it seems that they are not able to considerably improve on the simplicity of uncertainty sampling. It is very easy to implement and has been tested on many different datasets and domains over the years, **consistently improving on random sampling**. This means that if you don't have time to experiment with different methods, uncertainty sampling is a safe bet to use in your domain and will probably result in a labeled dataset which improves upon simple random sampling.

We should note that **the level of success of uncertainty sampling we report in this blog isn't reflected in some of the papers**. In both the core set paper and the adversarial approach's papers, uncertainty sampling is reported to only be on par with random sampling, sometimes being worse than it. This is not at all what we have seen, with uncertainty sampling consistently performing great with no need to fine tune any parameters or get confused with the implementation.

### Classification Isn't Everything
Finally, we saw that the active learning framework is inherently built for classification tasks. While there are applications of active learning for other types of tasks which we didn't cover here, in general it isn't trivial to adapt an active learning method to a completely new domain, and this may be one of the issues that prevents this field from making a stronger mark in machine learning industry. It is especially funny since **usually classification problems have data which is easier to label**, while the data that is harder to label and would benefit most from active learning comes from other domains.

This situation exactly is what makes the core set and DAL approaches so appealing. By using only the representation learned by the model on the original task and not the probability over labels, these methods are able be more task agnostic. Instead of requiring that the model we are learning is a classification model, all that is required is that it will **learn some representation of the data that is relevant to the task**, which is a much more reasonable assumption. This means that **these approaches open up active learning to many other machine learning tasks** and are a good option for applications where there isn't a good adaptation of other active learning methods.

We should say that the core set approach with the MIP formulation, which is considerably better than the greedy core set approach, is very computationally intensive and requires solving optimization tasks with many variables, which means it isn't a simple plug and play method like uncertainty sampling. This is where we think DAL can offer a great improvement compared to todays methods - it has the task-agnostic property of core set allowing it to transfer easily between domains, while being easier to implement and scale to large unlabeled datasets.

It would be interesting to see other, simpler task-agnostic methods pop up, since this kind of approach is probably the best bet at making active learning more common in industry applications.

## Summary
This brings our deep dive into active learning to a close. As we've seen, active learning is a highly developed and researched field with a clear motivation for improving the costs of data annotation, a big problem in many real-world machine learning applications.

The motivation of this field makes people come at it from many different directions, resulting in very cool and interesting ideas which could be applied to other problems as well. We saw ideas from classic optimization, Bayesian deep learning, adversarial examples, not to mention ideas we didn't cover which are detailed in the [references post][ref].

And that's it. We hope you've enjoyed this blog and gained a lot from it!

[ref]: https://dsgissin.github.io/DiscriminativeActiveLearning/2018/07/05/References.html

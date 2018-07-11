---
layout: post
title: Conclusion
---

In this blog series we've covered the framework of active learning from it's initial formulation and algorithms and up to today's state of the art methods. We tried to be concise but there was a lot to take in, so we'll detail the most important take home messages from all of this in our mind.

### The Data & Model Matter
We saw in trying to figure out what went wrong with EGL that it performed really poorly in our experiments, while it performed really well for the Baidu team on a speech recognition task. We conjectured that this was because of how the gradients behave differently for different models and machine learning tasks, and that EGL (and maybe the adversarial approach) could behave very differently on different tasks.

The thing is, **almost every research paper on modern active learning uses the same task, datasets and models** (image classification using some classic architecture). This isn't too surprising, since these datasets take a reasonable amount to time to train on and there is a lot of relevant open source code. But most people who would like to try active learning on an industry application won't use these datasets and probably won't tackle an image classification problem, and this means that **all of the comparisons from the literature could be irrelevant** for their scenario, especially with the differences between the methods being relatively small.

This means that if you want to try active learning in a new domain, choosing the state of the art method on MNIST isn't necessarily the right thing to do. If you have a good amount of labeled data to start with, it could be worth the time to **experiment with the different methods** like we did here. Otherwise, you should probably **avoid the methods which are more prone to change with the domain** (EGL in particular).

### The Batch Size Doesn't Matter That Much
even in high batch sizes which are more realistic to industry we get great results for greedy methods

### Nothing Really Beats Uncertainty Sampling
simple is good

### Classification Isn't Everything
adapting to other stuff is hard and that's why core set is good



- add a comment on the amount of methods and approaches that exist today, using many different tools in ML and optimization, and 

In the [next post][ref post] bla bla bla...
We hope you enjoyed bla bla bla...

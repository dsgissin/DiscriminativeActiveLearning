---
layout: post
title: Active Learning Review
comments: true
---

When we study supervised machine learning, the basic assumption usually is that we have a relatively large training set of labeled examples with which to train our model, and the hard part is to design the model or learning algorithm that will achieve the best results on unseen examples. This is especially true when we talk about deep learning models, where we assume that we have a huge training set that allows us to design deeper and deeper networks without suffering from overfitting.

In practice though, this isn't always the case. Not every learning task that we face has its own MNIST that we can easily download and play around with - most of the tasks that are interesting require data collection, cleaning and labeling. Even if you're one of the industry giants and have access to a huge amount of data, you'll still need to label that data. And labeling usually isn't as simple as choosing the class of the image either... You might need someone to place a bounding box around each interesting region of an image, or pay a linguist to pick the correct part-of-speech tag for every word in a corpus, or pay a radiologist a lot of money to go over many MRI scans and mark tumors.

So, while we would like to spend most of our time tweaking our hyper parameters to get the best results on our data, we end up spending most of it (along with a lot of our money) on labeling the data. This issue has been one of the key drives for research into unsupervised learning, semi-supervised learning and our topic today - Active Learning (AL).

### Blog Structure

In this blog we'll dive into active learning, starting with the basic framework and approaches and moving along to a more modern and practical setting. We will also evaluate and compare these methods in a controlled environment to see what really works (at least on the classic MNIST and CIFAR datasets). Finally, we will suggest a new method for active learning and detail the thought process that led us during the research in trying to get this new method to work well. 

All of the code for recreating the experiments in this blog, along with our implementations of the different methods are available in our [github repository][git] and you are free to use it and learn from it.

The blog posts are arranged as follows:

- [Introduction to Active Learning][intro post] - in this post we introduce the active learning framework and the classic algorithms developed for it.

- [Batch Active Learning][batch post] - in this post we extend the framework to a more realistic setting, and detail today's state of the art methods in this framework using deep learning models.

- [Comparing The Methods][compare post] - in this post we compare the state of the art methods against each other in the most impartial way we can, using the MNIST and CIFAR-10 datasets.

- [Discriminative Active Learning][dal post] - in this post we introduce the new active learning method we developed, tracing our research process from the initial idea through the experiments we've made. We also compare the method to existing methods.

- [Conclusion][conclusion post] - in this post we wrap everything up and try to focus on the most important take home messages from all of the things we saw in this blog series.

- [References][ref post] - if all of this isn't enough, in this post we give references to all of the methods and ideas covered in this post and more so you can keep learning about this interesting field.

We really hope you enjoy reading through this review of active learning and gain a lot out of it.

[git]: https://github.com/dsgissin/DiscriminativeActiveLearning
[intro post]: https://dsgissin.github.io/DiscriminativeActiveLearning/2018/07/05/AL-Intro.html
[batch post]: https://dsgissin.github.io/DiscriminativeActiveLearning/2018/07/05/Batch-AL.html
[compare post]: https://dsgissin.github.io/DiscriminativeActiveLearning/2018/07/05/Experiments.html
[dal post]: https://dsgissin.github.io/DiscriminativeActiveLearning/2018/07/05/DAL.html
[conclusion post]: https://dsgissin.github.io/DiscriminativeActiveLearning/2018/07/05/Conclusion.html
[ref post]: https://dsgissin.github.io/DiscriminativeActiveLearning/2018/07/05/References.html
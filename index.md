---
layout: page
title: Blog Intro
---

When we study supervised machine learning, the basic assumption usually is that we have a relatively large training set of labeled examples with which to train our model, and the hard part is to design the model or learning algorithm that will achieve the best results on unseen examples. This is especially true when we talk about deep learning models, where we assume that we have a huge training set that allows us to design deeper and deeper networks without suffering from overfitting.

In practice though, this isn't always the case. Not every learning task that we face has it's own MNIST that we can easilly download and play around with - most of the tasks that are interesting require data collection, cleaning and labeling. Even if you're one of the industry giants and have access to a huge amount of data, you'll still need to label that data. And labeling usually isn't as simple as choosing the class of the image either... You might need someone to place a bounding box around each interesting region of an image, or pay a linguist to pick the correct part-of-speech tag for every word in a corpus, or pay a radiologist a lot of money to go over many MRI scans and mark tumors.

So while we would like to spend most of our time tweaking our hyper parameters to get the best results on our data, we end up spending most of it (along with a lot of our money) on labeling the data. This issue has been one of the key drives for research into unsupervised learning, semi-supervised learning and our topic today - Active Learning.

In this blog we'll dive into active learning, starting with the basic framework and approaches and moving along to a more modern and practical setting. We will also evaluate and compare these methods in a controlled environment with [our own implementation][github] to see what really works (at least on the classic MNIST and CIFAR datasets). The blog posts are arranged as follows:

- 




[git]: https://github.com/dsgissin/DiscriminativeActiveLearning
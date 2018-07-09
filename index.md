---
layout: home
title: Active Learning Review
---

When we study supervised machine learning, the basic assumption usually is that we have a relatively large training set of labeled examples with which to train our model, and the hard part is to design the model or learning algorithm that will achieve the best results on unseen examples. This is especially true when we talk about deep learning models, where we assume that we have a huge training set that allows us to design deeper and deeper networks without suffering from overfitting.

In practice though, this isn't always the case. Not every learning task that we face has it's own MNIST that we can easilly download and play around with - most of the tasks that are interesting require data collection, cleaning and labeling. Even if you're one of the industry giants and have access to a huge amount of data, you'll still need to label that data. And labeling usually isn't as simple as choosing the class of the image either... You might need someone to place a bounding box around each interesting region of an image, or pay a linguist to pick the correct part-of-speech tag for every word in a corpus, or pay a radiologist a lot of money to go over many MRI scans and mark tumors.

So while we would like to spend most of our time tweaking our hyper parameters to get the best results on our data, we end up spending most of it (along with a lot of our money) on labeling the data. This issue has been one of the key drives for research into unsupervised learning, semi-supervised learning and our topic today - Active Learning (AL).

## Blog Structure

In this blog we'll dive into active learning, starting with the basic framework and approaches and moving along to a more modern and practical setting. We will also evaluate and compare these methods in a controlled environment with [our own implementation][git] to see what really works (at least on the classic MNIST and CIFAR datasets). The blog posts are arranged as follows:

- 

TODO:

- conclusions: as the batch size rises, Core-Set becomes more relevant. Also, the data type is very important and these experiments are only on image data...

- Core-Set is good because it isn't restricted to classification tasks

- note in the index file about the experiments and how we tried to make them fair (but that the literature isn't very fair) with a link to the detailing of the experiments and the choices we've made.

- note about the core set algorithm - as the batch size is smaller, the MIP solution is more important.

- Also note in the experiments section which algorithms use sub sampling for computational purposes (core set for instance)

- spell check all of the posts in a real text editor

- add in the references the papers of the methods that weren't mentioned (like the bandit formulation)

- add a comment on the amount of methods and approaches that exist today, using many different tools in ML and optimization, and hyperlink non obvious things.


[git]: https://github.com/dsgissin/DiscriminativeActiveLearning
---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: page
title: Intro
---

When we study supervised machine learning, the basic assumption usually is that we have a relatively large training set of labeled examples with which to train our model, and the hard part is to design the model or learning algorithm that will achieve the best results on unseen examples. This is especially true when we talk about deep learning models, where we assume that we have a huge training set that allows us to design deeper and deeper networks without suffering from overfitting.
In practice though, this isn't always the case. Not every learning task that we face has it's own MNIST that we can easilly download and play around with - most of the tasks that are interesting require data collection, cleaning and labeling. Even if you're one of the industry giants and have access to a huge amount of data, you'll still need to label that data. And labeling isn't always as simple as choosing the class of the image - you might need someone to place a bounding box around each interesting region of an image, or pay a linguist to pick the correct part of speech tag for every word in a corpus, or pay a radiologist a lot of money to go over many MRI scans and mark tumors.

Active learning is a framework 
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
</head>
<body>
  <h1>How Attention Improves a CNN’s performance: A Beginner's Perspective</h1>
  <h2>Tags:</h2>
  <p>Data Science, Computer Vision, Attention Layer, Artificial Intelligence, Machine Learning</p>
  <h2>Authors:</h2>
  <p>Ong Jun Hong, Heng Kim Rui, Aw Chong Kiang, Lum Nian Hua Eunice</p>
  
  <h2>Motivation</h2>

  <p align="justify">
In recent years, image classification has witnessed remarkable advancements with the advent of deep learning models. Among these models, convolutional neural networks (CNNs) have emerged as powerful tools for extracting hierarchical representations from images. The VGG16_bn model, with its deep architecture and batch normalization, has proven to be a reliable choice for various computer vision tasks, including image classification.However, despite the remarkable success of CNNs, they often treat all image regions equally and fail to focus on the most discriminative parts of an image. This limitation can hinder their performance, especially when dealing with complex and cluttered scenes. To address this, attention mechanisms have gained significant attention in the deep learning community.

The idea behind attention mechanisms is to enable the network to selectively focus on informative regions of an image while suppressing irrelevant or noisy regions. By incorporating attention layers into the VGG16_bn model, we can enhance its ability to attend to salient features and make more informed decisions during the classification process.

The motivation behind this article is to understand the attention mechanism and explore the effectiveness of attention mechanisms in improving the image classification performance of the VGG16_bn model. By introducing attention layers, we aim to enable the model to dynamically allocate its computational resources to the most relevant image regions, effectively capturing fine-grained details and improving its discriminative power.
  </p>

  <h2> What is Attention? </h2>
  <p align="justify">
Attention is a topic widely discussed and studied in both neuroscience and psychology. While there are many definitions of Attention, it can be considered as a resource allocation scheme - means to quickly select and process more important information from massive information using limited resources.

Attention was originally introduced as an extension to recurrent neural networks. With the introduction of the Transformer model, attention became popular and was quickly adopted for a variety of deep learning models across many different domains and tasks such as image processing, video processing, time-series dataset and recommender systems.
  </p>

  <h3> Attention Score Function </h3>
  <h3> Alignment Function </h3>  
  <h3> Type of Attention Mechanisms </h3>

  <h2> Model - VGG16 with Attention </h2>
  <p align="justify">
For the actual classification task, VGG16 with Attention layers will be used for demonstration based on the architecture first proposed in <a href="https://www2.cs.sfu.ca/~hamarneh/ecopy/ipmi2019.pdf?ref=blog.paperspace.com">this paper</a>.

  </p>
  
</body>
</html>

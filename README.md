# How Attention Improves a CNN’s performance: A Beginner's Perspective <a name="home"></a>
Data Science, Computer Vision, Attention Layer, Artificial Intelligence, Machine Learning 

Completed by: Ong Jun Hong, Heng Kim Rui, Aw Chong Kiang, Lum Nian Hua Eunice

## Table of Contents
1. [Introduction](#intro)
2. [What is Attention?](#what)
3. [Structure of a Basic Attention Model](#struc)
4. [Attention Score Function](#score)
5. [Attention Alignment Function](#align)


## [Introduction](#home) <a name="intro"></a>
In recent years, image classification has witnessed remarkable advancements with the advent of deep learning models. Among these models, convolutional neural networks (CNNs) have emerged as powerful tools for extracting hierarchical representations from images. The VGG16_bn model, with its deep architecture and batch normalization, has proven to be a reliable choice for various computer vision tasks, including image classification.

However, despite the remarkable success of CNNs, they often treat all image regions equally and fail to focus on the most discriminative parts of an image. This limitation can hinder their performance, especially when dealing with complex and cluttered scenes. To address this, attention mechanisms have gained significant attention in the deep learning community.

The idea behind attention mechanisms is to enable the network to selectively focus on informative regions of an image while suppressing irrelevant or noisy regions. By incorporating attention layers into the VGG16_bn model, we can enhance its ability to attend to salient features and make more informed decisions during the classification process.

The motivation behind this article is to understand the attention mechanism and explore the effectiveness of attention mechanisms in improving the image classification performance of the VGG16_bn model. By introducing attention layers, we aim to enable the model to dynamically allocate its computational resources to the most relevant image regions, effectively capturing fine-grained details and improving its discriminative power.

## [What is Attention?](#home) <a name="what"></a>
Attention is a topic widely discussed and studied in both neuroscience and psychology. While there are many definitions of Attention, it can be considered as a resource allocation scheme - means to quickly select and process more important information from massive information using limited resources.

Attention was originally introduced as an extension to recurrent neural networks. With the introduction of the Transformer model, attention became popular and was quickly adopted for a variety of deep learning models across many different domains and tasks such as image processing, video processing, time-series dataset and recommender systems.

## [Structure of a Basic Attention Model](#home) <a name="struc"></a>
[Figure 1](#fig1) shows an overview of attention model. While there are many variations of attention mechanism being employed, the objective of all attention models is to generate the context vector, which is usually a weighted average of all value vectors. This context vector will then be used to compute the prediction output. All attention models will need to have the following functions in order to output the context vector:

| Attention Function | Description |
| -- | -- |
| Score Function | &bull; Score function $score$ use query $q$ and keys matrix $K$ to calculate vector of attention scores $e = [e_1, \ldots, e_{n_f}] \in \mathbb{R}^{n_f}$ where $n_f$ represents the number of features that are extracted from inputs: $$e_l = score(q, k_l)$$ |
| Distribution Function <br>(Also known as alignment function) | &bull; Calculate the attention weights by redistributing the attention scores (which can generally have a wide range outside of [0, 1]) such that the attention weight is aligned to the correct value vector. <br>&bull; The vector of attention weights $a = [a_1, \ldots, a_{n_f}] \in \mathbb{R}^{n_f}$ is used to produce the context vector $c \in \mathbb{R}^{d_v}$ by calculating a weighted average of the columns of the values matrix $V$: $$c = \sum_{l=1}^{n_f} a_l * v_l$$ |

 ![attention model](./image/attention_model.png)  <a name="fig1"></a> 

Figure 1: Overview of Attention Model ([[3]](#3) Niu, Z. (2021) p.g. 3)

## [Attention Score Function](#home) <a name="score"></a>
As mentioned earlier, query symbolizes a request for information. Attention score represents how important the information contained in the key vector is according to the query. List of different types of score functions are shown below:

| Score Function | Description |
| -- | -- |
| Additive | &bull; Element wise summation of Weighted matrices of query and key followed by activation function <br>&bull; Britz et al. [[2]](#2) found that parameterized additive attention mechanisms outperformed multiplicative mechanisms slightly but consistently. $$w^T * act(W_1 * q + W_2 * k_l + b)$$ $$\text{where } w \in \mathbb{R}^{d_w}, W_1 \in \mathbb{R}^{d_w \times d_q}, W_2 \in \mathbb{R}^{d_w \times d_k}, b \in \mathbb{R}^{d_w}$$ |
| Concat | &bull; Instead of having 2 weights matrices for q and k, q and k are concatenated; and a single weight is applied to it. $$w^T * act(W[q; k] + b)$$ $$\text{where } w \in \mathbb{R}^{d_w}, W \in \mathbb{R}^{d_w \times d_q+d_k}, b \in \mathbb{R}^{d_w}$$ |
| Multiplicative <br>(Dot-Product) | &bull; Computationally inexpensive due to highly optimized vector operations. <br>&bull; May produce non-optimal results when dimension dk  is too large i.e. softmax of these large numbers will result in gradients becoming too small, causing trouble of model converging. $$q^T \cdot k_l$$ |
| Scaled Multiplicative | &bull; Address the issue dimension $d_k$ being too large. $$\frac{{q^T \cdot k_l}}{{\sqrt{d_k}}}$$ |
| General | &bull; Extend multiplicative function by introducing weights matrix W, which can be applied to keys and queries with different representation. $$k^T_l * W * q$$ $$\text{where } W \in \mathbb{R}^{d_k \times d_q}$$ |
| Biased General | &bull; Further extension of general function by including a bias weight vector. $$k^T_l * W * q + b$$ $$\text{where } W \in \mathbb{R}^{d_k \times d_q}, b \in \mathbb{R}^{d_k}$$ |
| Activated General | &bull; Includes both bias and activation function, $act$. $$act(k^T_l * W * q + b)$$ $$\text{where } W \in \mathbb{R}^{d_k \times d_q}, b \in \mathbb{R}^{d_k}$$ |
| Similarity | &bull; Weightages are calculated based on how ‘similar’ are the key and query vectors such as using Euclidean ($L2-norm$) distance and cosine similarity. |

Note that:
- $k$ (vector; an element of $K$ matrix), $v$, $b$, $W$, $W_1$ and $W_2$ are learnable parameters. 
- $d_k$ is the dimension of key matrix, $K$.
- $act$ is the nonlinear activation function such as tanh and ReLU.

There is no specific score function that can be used across domains. Choice of score function for a particular task is often based on empirical experiments. However, if efficiency is vital, multiplicative or scaled multiplicative core functions are typically the best choice.

## [Attention Alignment Function](#home) <a name="align"></a>
The goal of attention alignment is to generate the context vector, which will be used by the output model to generate prediction. Following steps are taken:
1. Using the attention scores as input, it calculates the attention weights for each corresponding value vector in $V$ matrix.
2. These attention weights can then be used to create the context vector $c$ by, for example, taking the weighted average of the value vectors. Examples of alignment functions are as follows:


## [References](#home) <a name="ref"></a>
[1] <a name='1'></a> Yan, Y., Kawahara, J., & Hamarneh, G. (2019). Melanoma Recognition via Visual Attention. In Lecture Notes in Computer Science <br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (pp. 793–804). Springer Science+Business Media. https://doi.org/10.1007/978-3-030-20351-1_62
<br> [2] <a name='2'></a> Britz, D., Goldie, A., Luong, M., & Le, Q. V. (2017). Massive Exploration of Neural Machine Translation Architectures. <br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;https://doi.org/10.18653/v1/d17-1151
<br> [3] <a name='3'></a> Niu, Z., Zhong, G., & Yu, H. (2021). A review on the attention mechanism of deep learning. Neurocomputing, 452, 48–62. <br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;https://doi.org/10.1016/j.neucom.2021.03.091
<br> [4] <a name='4'></a> Brauwers, G., & Frasincar, F. (2023). A General Survey on Attention Mechanisms in Deep Learning. IEEE Transactions on Knowledge and <br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Data Engineering, 35(4), 3279–3298. https://doi.org/10.1109/tkde.2021.3126456

# How Attention Improves a CNN’s performance: A Beginner's Perspective <a name="home"></a>
**Index Terms**: Data Science, Computer Vision, Attention Layer, Artificial Intelligence, Machine Learning 

Completed by: Ong Jun Hong, Heng Kim Rui, Aw Chong Kiang, Lum Nian Hua Eunice

## Table of Contents
1. [Introduction](#intro)
2. [What is Attention?](#what)
3. [Structure of a Basic Attention Model](#struc)
4. [Attention Score Function](#score)
5. [Attention Alignment Function](#align)
6. [Types of Attention Mechanism](#type)
7. [Model Architecture](#model)
8. [Experiment](#exp)
9. [Results](#result)
10. [Conclusion](#conclude)
11. [Comparison with GradCam](#compare)
12. [References](#ref)

## [Introduction](#home) <a name="intro"></a>
In recent years, image classification has witnessed remarkable advancements with the advent of deep learning models. Among these models, convolutional neural networks (CNNs) have emerged as powerful tools for extracting hierarchical representations from images. The VGG16_bn model, with its deep architecture and batch normalization, has proven to be a reliable choice for various computer vision tasks, including image classification.

However, despite the remarkable success of CNNs, they often treat all image regions equally and fail to focus on the most discriminative parts of an image. This limitation can hinder their performance, especially when dealing with complex and cluttered scenes. To address this, attention mechanisms have gained significant attention in the deep learning community.

The idea behind attention mechanisms is to enable the network to selectively focus on informative regions of an image while suppressing irrelevant or noisy regions. By incorporating attention layers into the VGG16_bn model, we can enhance its ability to attend to salient features and make more informed decisions during the classification process.

The motivation behind this article is to understand the attention mechanism and explore the effectiveness of attention mechanisms in improving the image classification performance of the VGG16_bn model. By introducing attention layers, we aim to enable the model to dynamically allocate its computational resources to the most relevant image regions, effectively capturing fine-grained details and improving its discriminative power.

## [What is Attention?](#home) <a name="what"></a>
Attention is a topic widely discussed and studied in both neuroscience and psychology. While there are many definitions of Attention, it can be considered as a resource allocation scheme - means to quickly select and process more important information from massive information using limited resources.

Attention was originally introduced as an extension to recurrent neural networks. With the introduction of the Transformer model, attention became popular and was quickly adopted for a variety of deep learning models across many different domains and tasks such as image processing, video processing, time-series dataset and recommender systems.

In the context of image processing, attention tells the model to perform the following:
- Focus on local regions within the same object e.g. face, eyes, nose, etc.
- Differentiate between different objects e.g. dog versus wolf.
- Differentiate between the object and the background e.g. wolf in snowy background.

## [Structure of a Basic Attention Model](#home) <a name="struc"></a>
[Figure 1](#fig1) shows an overview of attention model. While there are many variations of attention mechanism being employed, the objective of all attention models is to generate the context vector, which is usually a weighted average of all value vectors. This context vector will then be used to compute the prediction output. All attention models will need to have the following functions in order to output the context vector:

| Attention Function | Description |
| -- | -- |
| Score Function | &bull; Score function $score$ use query $q$ and keys matrix $K$ to calculate vector of attention scores $e = [e_1, \ldots, e_{n_f}] \in \mathbb{R}^{n_f}$ where $n_f$ represents the number of features that are extracted from inputs: $$e_l = score(q, k_l)$$ |
| Distribution Function <br>(Also known as alignment function) | &bull; Calculate the attention weights by redistributing the attention scores (which can generally have a wide range outside of [0, 1]) such that the attention weight is aligned to the correct value vector. <br>&bull; The vector of attention weights $a = [a_1, \ldots, a_{n_f}] \in \mathbb{R}^{n_f}$ is used to produce the context vector $c \in \mathbb{R}^{d_v}$ via $\phi(a_i, v_i)$ function, which is usually the weighted average of the columns of the values matrix $V$: $$c = \sum_{l=1}^{n_f} a_l * v_l$$ |

 ![attention model](./image/attention_model.png)  <a name="fig1"></a> 

Figure 1: Overview of Attention Model ([[3]](#3) Niu, Z. (2021) p.g. 3)

## [Attention Score Function](#home) <a name="score"></a>
As mentioned earlier, query symbolizes a request for information. Attention score represents how important the information contained in the key vector is according to the query. List of different types of score functions are shown below:

| Score Function | Description |
| -- | -- |
| Additive | &bull; Element wise summation of Weighted matrices of query and key followed by activation function <br>&bull; Britz et al. [[2]](#2) found that parameterized additive attention mechanisms slightly but consistently outperformed multiplicative mechanisms. $$w^T * act(W_1 * q + W_2 * k_l + b)$$ $$\text{where } w \in \mathbb{R}^{d_w}, W_1 \in \mathbb{R}^{d_w \times d_q}, W_2 \in \mathbb{R}^{d_w \times d_k}, b \in \mathbb{R}^{d_w}$$ |
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
2. These attention weights can then be used to create the context vector $c$ by, for example, taking the weighted average of the value vectors.

Examples of alignment functions are as follows:

| Alignment Function | Description |
| -- | -- |
| Softmax | &bull; Most popular alignment method to calculate attention weights. <br>&bull; Often referred to as soft alignment in computer vision or global alignment for sequence data. <br>&bull; Ensures that every part of input receives at least some amount of attention. <br>&bull; Introduce probabilistic interpretation to input vectors, allowing easy analysis of which parts of inputs are important to the output predictions. |
| Sparsemax | &bull; Assign exactly zero probability to some of its output variables if sparse probability distribution is desired. |
| Sigmoid | &bull; Scaled energy scores between 0 and 1 like softmax. <br>&bull; However, sum of all attention weights will not be 1 for multiple features. |
| Hard Alignment | &bull; Forces attention model to focus on exactly one feature vector.<br>&bull; Applies softmax on the attention scores but uses the outputs as probabilities to draw the choice of the one value vector instead of weighted averages of all value vectors.<br>&bull; While it is more efficient at inference compared to soft alignment, gradients are not differentiable. <br>&bull; As such, training cannot be done via regular backpropagation. Instead, sampling or reinforcement learning are required to calculate the gradient at the hard attention layer. |
| Local Alignment | &bull; Applies softmax distribution on a subset of inputs rather than the entire inputs.<br>&bull; First predicts a single aligned position $p_t$ for the current target word.<br>&bull; Then calculate the context vector $c$ based on window centered around the source position $p_t$ i.e. $[-D+p_t, D+p_t]$. <br>&bull; The advantage is that the gradient is differentiable despite taking only a subset of inputs to perform softmax each time. |
| Reinforced Alignment | &bull; Uses reinforcement learning agent, similar to hard alignment, to choose a subset of feature vectors.<br>&bull; However, the attention calculation based on these chosen feature vectors is the same as regular soft alignment i.e. allows back propagation. |

## [Types of Attention Mechanism](#home) <a name="type"></a>
Brauwers et al. [[4]](#4) created a taxonomy to classify the different types of attention mechanisms into 3 main categories namely: feature-related, query-related or general (i.e. not feature or query related).

1. General
- Consists of attention mechanisms (i.e. attention scoring, attention alignment and attention dimensionality) that can be applied in any type of attention model.
- Different attention scoring and alignment have been covered in Section 4: [Attention Score Function](#score) and Section 5: [Attention Alignment Function](#end).
- Attention dimensionality is choosing between a single attention score and weight for the entire feature vector or calculating weights for every single element (entire dimension)  of that specified feature vector.
   
3. Feature-Related

| Feature Type | Description |
| -- | -- |
| Number of inputs to be attended | &bull; E.g. Co-Attention to jointly attend to both an image and a question (i.e. 2 inputs). <br>&bull; Rotary attention incorporates 3 input phrases: left, right and target phrase. |
| Different levels of details | &bull; E.g. attention-via-attention predict sentence translation character-by-character while also incorporating information from a word-level attention module (i.e. 2 levels). <br>&bull; Hierarchical attention starts at lowest level and then creates representation of next level using attention until highest level is reached. E.g. words -> sentences -> documents. |
| Single or multiple representation of inputs | &bull; E.g. multiple representations of the same book can be textual, syntactic, semantic, visual, etc. <br>&bull; Multi-representational attention takes weighted average of multiple representation, where the weights are determined by attention. |
   
3. Query-Related

| Query Type | Description |
| -- | -- |
| Basic Query | &bull;  Straightforward to define based on data and model <br>&bull; E.g. patient characteristics can be a query for medical image classification. |
| Specialized Query | &bull; E.g. rotary attention uses context vector from another attention module as query. <br>&bull; On the hand, interactive co-attention uses an averaged keys vector based on another input as query. |
| Self-attention (Intra-attention) | &bull; Often used in feature model to create improved represenations of the feature vectors by: <br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Setting feature vectors to be equal to the acquired self-attention context vectors. <br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Adding the context vectors to the previous feature vectors with an additional normalization layer. <br>&bull; Query is ignored/removed and only the key is used to calculate the attention score. <br>&bull; Using Concat scoring function, the equation would be: $$score(k_l) = w^T * act(W * k_l + b)$$ |
| Multiple or single query | &bull; Multi-head attention processes multiple attention modules in parallel. <br>&bull; Multi-hop attention to refine the context vectors iteratively through the multiple attention modules. <br>&bull; Capsule-based attention which assigns a separate attention module to each of the prediction classes. |

## [Model Architecture](#home) <a name="model"></a>
We leverage on the attention-based method for melanoma recognition proposed by Yan, Y. [[1]](#1) for image classification of cats and dogs. The overall network architecture is shown in [Figure 2](#fig2).

![Overall Network Architecture](./image/VGG16_adapted.png) <a name="fig2"></a>

Figure 2: Overall Network Architecture ([[1]](#1) Yan, Y., J., & Hamarneh, G. (2019) p.g. 3)

Breakdown of the architecture is as follows:

- VGG-16 (yellow and red blocks), without the dense layers, serves as the backbone network.
- Pool-3 and pool-4 are intermediate feature maps found in the VGG-16 layers, while pool-5 is the final output of VGG-16 convolutional layers (i.e. without the dense players). 
- An attention module (gray block) is applied to pool-3 and pool-5 while the other is applied to pool 4 (closer to the output) and pool 5.
- Global average pooling is applied to outputs of the 2 attention module and pool-5 to generate 3 feature vectors (green blocks).
- These feature vectors are then concatenated together to form the final feature vector, which serves as the input to the classification layer.
- Classification layer (not shown above) is a fully connected dense layer to perform the classification.
- Whole network is trained end-to-end.

![Inner Workings of Attention Module](./image/attention_block_adapted.png) <a name="fig3"></a>

Figure 3: Inner Workings of Attention Module ([[1]](#1) Yan, Y., J., & Hamarneh, G. (2019) p.g. 4)

The inner workings of the attention modules (as shown in [Figure 3](#fig3)) are as follows:

- Being the last stage feature, output of pool-5 contains the most compressed and abstracted information over the entire image; and therefore serves a form of “global guidance” i.e. global feature $\mathcal{G}$.
- Upsampling of $\mathcal{G}$ via bilinear interpolation to ensure its spatial size is aligned with the intermediate features $\mathcal{F}$.
- $\mathcal{F}$ and upsampled $\mathcal{G}$ are fed through the attention module to generate a one-channel response $\mathcal{R}$: $$\mathcal{R} = W \circledast ReLU(W_f \circledast \mathcal{F} + up(W_g \circledast \mathcal{G}))$$ $$\text{where } \circledast \text{represents a convolutional operation; } W_f, W_g \text{ and } W \text{ are convolutional kernels; and } up(.) \text{ is a bilinear interpolation.}$$
- $\mathcal{R}$ is then transformed via $Sigmoid$ function to generate the attention map $\mathcal{A}$: $$\mathcal{A} = Sigmoid(\mathcal{R})$$
- Attention-version of pool-3 and pool-4 features are generated via ‘pixel-wise’ multiplication of the intermediate features with the attention map: $$\hat{f_i} = a_i \cdot f_i$$ $$\text{where scalar element } a_i \in \mathcal{A} \text{ represents the degree of attention to the corresponding spatial feature vector in } \mathcal{F} \text{;}$$ $$\text{and each feature vector } f_i \text{ is multiplied by the attention element } a_i \text{ to get the attention-version } \hat{f_i}$$

## [Experiments](#home) <a name="exp"></a>
We ran our experiments on 10000 images belonging to cats and dogs. No processing of images is required as the images are of similar size. No regularization via regions of interest are carried out. We didn't add any dropout.Hence we used just the binary cross-entropy loss instead of focal loss in the original literature without regularization terms.

PyTorch is used to implement the model. Back-bone network is initialized with VGG-16 pre-trained on ImageNet, and the attention modules are initialized with He’s initialization. Due to time constraints, the whole network is only trained end-to-end for 15 epochs using Adam optimizer. Working codes can be found in our jupyter notebook: _Notebook_Image Classification with Attention.ipynb_.

In terms of quantitative evaluation, the performance of the models with attention is measured using the average precision (AP) and the area under the ROC curve (AUC). These metrics were the official evaluation metrics used in the ISIC 2016 and 2017 challenges, respectively. The results of the experiments are then compared with VGG-16 network without attention mechanisms to identify whether attention does improve our image classification of dogs and cats. The original literature uses Sigmoid normalization for the attention mechanism. We tried out Softmax normalization as an alternative.

## [Results](#home) <a name="result"></a>
We will present the attention maps from Pool 3 and Pool 4 for Sigmoid and Softmax normalization.

1. Attention map from Pool 3 and Pool 4 using Sigmoid normalization



Hypertuning - Learning Rate

Learning Rate = 0.01
For the model using concat method, 
![image](https://github.com/krheng14/Image-Classification-with-Attention/assets/137394373/451b747f-3453-4b3f-b280-577f05650293)

![image](https://github.com/krheng14/Image-Classification-with-Attention/assets/137394373/c415a07c-f4b4-43d6-9d26-af91fa39f2f8)


Plotting out the attention layers, we can see that the attention layers can only pick out very small regions. Seems like the model is not able to optimise and converge in the loss function properly (learning rate too large).
![image](https://github.com/krheng14/Image-Classification-with-Attention/assets/137394373/af088026-5414-4986-978d-91d24523bf65)

It appears that the model is still unsure where to look at, to distinguish between cats and dogs.



Let us try a smaller learning rate = 0.0001
![image](https://github.com/krheng14/Image-Classification-with-Attention/assets/137394373/84e86d9e-b2da-4b03-b2f2-e95252352be5)

![image](https://github.com/krheng14/Image-Classification-with-Attention/assets/137394373/32d75800-53b0-4073-80ca-e3e5dfbaf13c)
The models are performing much better than before.

And the attention layers matrix.
![image](https://github.com/krheng14/Image-Classification-with-Attention/assets/137394373/a871284d-ac78-48ed-9554-9d5f1b8fa945)
1. It is a good idea for learning rate to be smaller.
2. It seems that the model is looking at 'eyes' to distinguish between cats and dogs.

We decided to try a different method, dot product attention layer, with learning rate of 0.0001

![image](https://github.com/krheng14/Image-Classification-with-Attention/assets/137394373/0b22de00-ad18-4e3c-bc2c-0a93746fa265)

It seems that the learning area is much smaller.
![image](https://github.com/krheng14/Image-Classification-with-Attention/assets/137394373/5ec37b01-5225-40be-b169-54cf1cb10dcd)

## [Comparison with GradCam](#compare) <a name="compare"></a>

Grad-cam is another method of deriving which part of the image is most relevant for making predictions.  
It passes an image through a layer in the model to get the activations of that layer and calculates the gradients of the output with respect to the activations of that layer. 
Grad-cam map is computed from combining the activations and gradients to get a weighted map that represents importance of different regions of image. 
However, it is a post-hoc method that evaluates the model after it is trained; it is unable to change the way the model learns, or change what the model learns. 
This is unlike attention, where we are jointly learning these attentional weights with the rest of the parameters in CNN, and these attentional weights will in turn helps the CNN model to predict better.

## [Conclusion](#home) <a name="conclude"></a>
We have explored the theoretical aspects of attention mechanisms, discussing different types of attention, and how attention mechanisms can be integrated into the VGG16_bn model architecture. Additionally, we visualized the impact of attention on model interpretability and visualized the attended regions to gain insights into the decision-making process. Through experiments on cats and dogs dataset, we have demonstrated the effectiveness of the attention-enhanced VGG16_bn model compared to the baseline model.

## [References](#home) <a name="ref"></a>
[1] <a name='1'></a> Yan, Y., Kawahara, J., & Hamarneh, G. (2019). Melanoma Recognition via Visual Attention. In Lecture Notes in Computer Science <br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (pp. 793–804). Springer Science+Business Media. https://doi.org/10.1007/978-3-030-20351-1_62
<br> [2] <a name='2'></a> Britz, D., Goldie, A., Luong, M., & Le, Q. V. (2017). Massive Exploration of Neural Machine Translation Architectures. <br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;https://doi.org/10.18653/v1/d17-1151
<br> [3] <a name='3'></a> Niu, Z., Zhong, G., & Yu, H. (2021). A review on the attention mechanism of deep learning. Neurocomputing, 452, 48–62. <br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;https://doi.org/10.1016/j.neucom.2021.03.091
<br> [4] <a name='4'></a> Brauwers, G., & Frasincar, F. (2023). A General Survey on Attention Mechanisms in Deep Learning. IEEE Transactions on Knowledge and <br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Data Engineering, 35(4), 3279–3298. https://doi.org/10.1109/tkde.2021.3126456

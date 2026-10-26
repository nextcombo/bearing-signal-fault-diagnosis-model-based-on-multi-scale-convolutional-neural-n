# Bearing Signal Fault Diagnosis Model Based on Multi-Scale Convolutional Neural Network (MSCNN-1D)

Preface
In modern industrial production, the fault diagnosis of mechanical equipment is crucial for ensuring production safety and extending equipment lifespan. Bearings, as key components of rotating machinery, are particularly important for early fault detection. Addressing the issue that traditional diagnostic methods heavily rely on signal feature extraction and have limited accuracy, this session implements a 1D signal processing model based on a multi-scale convolutional neural network framework in Python-Pytorch, called MSCNN-1D, to achieve efficient and accurate diagnosis of bearing vibration signals. This model updates the entire family of bearing fault diagnosis tools, and students are advised to update it according to the video tutorials for learning!
Dataset: Casio Westridge University (CWRU) Bearing Dataset
Environment Framework: Python 3.11, PyTorch 2.1 or higher versions can be run
Accuracy: Test set achieves 99% accuracy
Target Audience: Beginners, those requiring papers, and those pursuing graduation projects
Code Guarantees: The code is detailed with comments, ready to run without modification.
Introduction to Multi-Scale Convolutional Neural Networks
1.1 Model Overview
Multi-Scale Convolutional Neural Networks (MSCNN) are a type of deep learning structure that builds upon Convolutional Neural Networks (CNN). They design multiple convolution kernel sizes to extract features simultaneously, capturing multi-level and multi-granularity information from the input data, thereby enhancing the model's expressiveness and generalization performance.
1. Core Mechanism:
Traditional CNNs typically employ fixed-size convolutional kernels (e.g., 3×3 or 5×5) for feature extraction, which struggles to balance local details with broader contextual information simultaneously. By contrast, multi-scale convolution leverages the parallel setup of various different sizes of convolutional kernels (e.g., 3×3, 5×5, 7×7), performing multi-scale convolution operations on the same input feature map to obtain representations with varying receptive fields. Subsequently, these multi-scale features are fused (e.g., concatenated or weighted summation), achieving comprehensive information utilization.

1.2 Network Architecture Introduction
The Multi-Scale Convolutional Neural Network (MSCNN-1D) is a deep learning model specifically designed for one-dimensional signals such as bearing vibration signals, aiming to fully utilize multi-scale convolution to capture features at different frequencies and durations in the signal, thereby achieving precise fault diagnosis:
(1) Input layer
The model receives a one-dimensional multi-channel input signal, commonly a vibration sampling sequence, in the shape of (batch_size, channels, sequence_length). For example, data from single-channel or multi-channel sensors.
(2) Initial Convolutional Layer
Larger convolutional kernels (e.g., kernel size 9) are used to preprocess the raw input signal for feature extraction, extracting basic local information, with multiple output channels (e.g., 64 channels), and accompanied by ReLU activation functions to enhance nonlinear expressive capabilities.
(3) Multi-Scale Convolutional Block (MSB)
The core module MSB consists of several parallel convolutional branches, each using different sizes of convolutional kernels (e.g., 3, 5, 7), conducting convolution calculations on the same input feature map. This collects features from various scales (short-term, medium-term, long-term) to capture signal patterns from different scales. The output of MSB is the concatenation of multi-scale convolution results in channel dimensions, realizing rich representation fusion.
(4) Pooling Layer
After the multi-scale convolution output, a pooling layer (e.g., max pooling) is applied, down-sampling the length of the temporal signal while reducing computational load and extracting important features.
(5) MSB Stacking and MSB_mini Modules
The model is stacked with multiple MSB layers and a simplified MSB_mini module layer by layer, extracting signal features progressively to enhance deep expression capabilities. The MSB_mini generally employs simpler convolutional kernel combinations to reduce computational complexity while maintaining multi-scale analysis characteristics.

(6) 1×1 Convolution Fusion Layer
After multi-scale and multi-layer feature extraction, a 1×1 convolution layer is used to adjust and fuse the channel dimensions, reducing parameters and strengthening nonlinear combinations between channels, thereby improving compactness and discriminative power of feature representation.

(7) 1×1 Convolution Fusion Layer
Finally, a 1×1 convolution mapping is applied to the desired number of classification channels (e.g., number of fault categories), outputting corresponding category probabilities or logits for fault diagnosis classification.
MSCNN-1D models are compact and progressively fused through multi-scale convolution blocks, suitable for processing complex time-series signals, especially mechanical vibration signals, achieving high accuracy and real-time fault diagnosis, providing high-performance fault early warning technical support for intelligent manufacturing.

2 Data Preprocessing for CWRU Data
2.1 Import Data
Referencing previous articles, preprocessing for fault classification in 10 categories is performed, using the Caterpillar University bearing data 10-category dataset:

2.2 Data Preprocessing
A sliding window is created to make datasets, and the training set, validation set, and test set are uniformly divided into 7:2:1 ratios, finally saving the data.
3 Fault Diagnosis Model Based on MSCNN
3.1 Define the network model, set parameters, and train the model
50 epochs, accuracy 99%, the MSCNN classification effect is significant with multi-scale convolutional neural network MSCNN, which can extract important features from fault signal characteristics for model recognition, showing significant advantages:
Multi-scale response capability: different size convolutional kernels take care of both high-frequency and low-frequency information in the signal.
1. Automatic feature learning: Bypassing the complex manual feature extraction process, achieving end-to-end diagnosis.
2. Lightweight and efficient: Designing convolutional channels and layers reasonably to balance performance and computational efficiency.
3. Compatibility with multi-channel signals: Flexibly supporting data fusion from multiple sensors.
3.2 Model evaluation
(1) Accuracy, Precision, Recall, F1 Score
(2) Fault classification confusion matrix for 10 categories:
3. Update download
(1) 27 Bearing Fault Diagnosis Model | The most cost-effective comprehensive package across the network
27 Bearing Fault Diagnosis Model | The most cost-effective comprehensive package across the network
Previously interesting content:
Python - Kaiser Permanente University (CWRU) Bearing Data Interpretation and Classification Process
Python Bearing Fault Diagnosis (Part I): Short-time Fourier Transform STFT
Python Bearing Fault Diagnosis (Part II): Continuous Wavelet Transform CWT
Python Bearing Fault Diagnosis (Part III): Empirical Mode Decomposition EMD
Pytorch-LSTM Bearing Fault One-Dimensional Signal Classification (Part I)
Pytorch-CNN Bearing Fault One-Dimensional Signal Classification (Part II)
Pytorch-Transformer Bearing Fault One-Dimensional Signal Classification (Part III)
Python Bearing Fault Diagnosis (Part IV): Based on EMD-CNN Fault Classification
Python Bearing Fault Diagnosis (Part V): Based on EMD-LSTM Fault Classification
Python Bearing Fault Diagnosis (Part VI): Based on EMD-Transformer Fault Classification
Python Bearing Fault Diagnosis (Part VII): Based on EMD-CNN-LSTM Fault Classification
Python Bearing Fault Diagnosis (Part VIII): Based on EMD-CNN-GRU Parallel Model Fault Classification
Based on FFT + CNN - BiGRU-Attention: Fusion of time-domain and frequency-domain features attention in bearing fault identification model based on FFT + CNN
Bearing Fault Identification Model Based on FFT + CNN - Transformer Time-Domain and Frequency-Domain Feature Fusion
Python Bearing Fault Diagnosis (IX) Fault Classification Based on VMD+CNN-BiLSTM
Python Bearing Fault Diagnosis (10) Failure Classification Based on VMD+CNN-Transformer
Python Bearing Fault Diagnosis (11) Based on VMD+CNN-BiGRU-Attenion for Fault Classification
Cross Attention FFT + CNN - BiLSTM-CrossAttention Bearing Fault Identification Model
Cross Attention Fusion of Time-Domain and Frequency-Domain Features for Bearing Failure Identification Model with FFT + CNN-Transformer-CrossAttention
Python Bearing Fault Diagnosis Beginner Tutorial
Python Bearing Fault Diagnosis (13) Super-strong Machine Learning Recognition Model Based on Fault Signal Feature Extraction
Python Bearing Fault Diagnosis (14) High-Innovation Fault Identification Model
Python Bearing Fault Diagnosis (15) One-Dimensional Fault Signal Recognition Model Based on CNN-Transformer
Python Bearing Fault Diagnosis (16) High-Innovation Fault Identification Model (II)
| Sci 1 Zone High-Innovation Bearing Fault Diagnosis Model!
Python Bearing Fault Diagnosis (18) - Innovative Diagnosis Model Based on CNN-TCN-Attention
High-Innovation Bearing Fault Diagnosis Model Based on GADF+Swin-CNN-GAM
Attention Magic Mod | Superb Bearing Fault Diagnosis Model!
Bearing Faults—Cross Attention Innovative Model Full Set
Python Bearing Fault Diagnosis (19) - Innovation in Diagnosis Model Based on Transformer-BiLSTM
END  
2. Friends who enjoy this can click to follow, give the article a "like", share on social media or discuss in groups.

## Images

![Bearing Signal Fault Diagnosis Model Based on Multi-Scale Convolutional Neural Network (MSCNN-1D)](11.jpg)

<!-- PAYMENT_BLOCK -->

## Get this dataset

- Payment link (Stripe): [Buy now](https://buy.stripe.com/3cs8yP7sY87d0vu9AB)
- Payment QR code: ![Payment QR code](pay.png)

<!-- /PAYMENT_BLOCK -->

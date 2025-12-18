# PathTransGCN
## Introduction
In this study, we propose a multi-omics integration model based on pathway self-attention and Graph Convolutional Network (GCN) — named PathTransGCN, which incorporates cancer-specific pathway information into multi-omics data analysis with the aim of improving the accuracy of complex disease classification. 
![模型](https://github.com/user-attachments/assets/1b3e4918-becc-452f-8da3-896972aadb96)
The input of PathTransGCN is multi-omics data. After preprocessing the data, embedding learning is performed in conjunction with the pathway self-attention module (details of this module are shown in Figure 1B). Meanwhile, a sample similarity network is constructed using the Similarity Network Fusion (SNF) framework. Finally, the embedded representations and sample adjacency matrix are input into the Graph Convolutional Network for training, and classification results are output.

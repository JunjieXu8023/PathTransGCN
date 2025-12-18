# PathTransGCN
## Introduction
In this study, we propose a multi-omics integration model based on pathway self-attention and Graph Convolutional Network (GCN) — named PathTransGCN, which incorporates cancer-specific pathway information into multi-omics data analysis with the aim of improving the accuracy of complex disease classification. 
![模型](https://github.com/user-attachments/assets/1b3e4918-becc-452f-8da3-896972aadb96)
The input of PathTransGCN is multi-omics data. After preprocessing the data, embedding learning is performed in conjunction with the pathway self-attention module (details of this module are shown in Figure 1B). Meanwhile, a sample similarity network is constructed using the Similarity Network Fusion (SNF) framework. Finally, the embedded representations and sample adjacency matrix are input into the Graph Convolutional Network for training, and classification results are output.
## Usage
The whole workflow is divided into three steps:

- Use PSAM to reduce the dimensionality of multi-omics data to obtain multi-omics feature matrix
- Use SNF to construct patient similarity network
- Input multi-omics feature matrix and the patient similarity network to GCN

## Requirements
- Python 3.8 or above
- Pytorch 2.1.0 or above
- pandas 1.5.3 or above
- snfpy 0.2.2
- scikit-learn 1.3.2

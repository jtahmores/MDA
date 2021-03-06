# MDA
Multi-source domain adaptation for image classification

This package contains the data and code used in multi-source domain adaptation for image classification (MDA), which is published at Machine Vision and Applications (2020).
You can download the paper from: "https://doi.org/10.1007/s00138-020-01093-2". 

# Motivation

In recent years, domain adaptation and transfer learning are known as promising techniques with admirable performance to deal with problems with distribution difference between the training (source domain) and test (target domain) data. In this paper, a novel unsupervised multi-source transductive transfer learning approach, referred to as multi-source domain adaptation for image classification (MDA), is proposed, to transfer knowledge across the selected samples of multiple-source domains and samples of target domain into a shared low-dimensional subspace with maximum decision regions.MDAextends maximum mean discrepancy criteria across multiple-source domains to find an optimal projection subspace and constructs embedded condensed domain-invariant clusters. Furthermore, MDA minimizes empirical risk and maximizes the rate of consistency between manifold and prediction function via learning an optimal classification. Extensive evaluations on two types of visual benchmark datasets under different difficulties illustrate that MDA significantly outperforms other baseline and state-of-the-art methods in both multiple- and single-source tasks.

# RUN

The original code is implemented using Matlab R2018a. For running the code, run the "demo_MDA_office_ThreeSrc.m" and "demo_MDA_PIE_ThreeSrc.m " files.

# Datasets

*_SURF_L10.mat:    features and labels related to Office-Caltech-10

PIE*.mat:    features and labels related to PIE dataset


# Reference

Karimpour, M., Noori Saray, S., Tahmoresnezhad, J. et al. Multi-source domain adaptation for image classification. Machine Vision and Applications 31, 44 (2020). https://doi.org/10.1007/s00138-020-01093-2
		
# Contact

Jafar Tahmoresnezhad (tahmores@gmail.com)
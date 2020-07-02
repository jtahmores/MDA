This package contains the data and code used in [1]. 

10 common categories are picked out from 4 datasets of object images: 
caltech-256 [2], amazon, webcam, and dslr [3]. We then follow the 
previously reported protocols for preparing features [3], i.e., extracting 
SURF features and quantizing them into an 800-bin histogram with codebooks 
computed via K-means on a subset of images from amazon. 

Data:
*_SURF_L10.mat:    features and labels related to Office-Caltech-10
PIE*.mat: features and labels related to PIE dataset

Code: code for the multi-source domain adaptation for image classification

The original code is implemented using Matlab R2018a. For running the code, run the "demo_MDA_office_ThreeSrc.m" and "demo_MDA_PIE_ThreeSrc.m " files.
            

------------
[1] Karimpour, M., Noori Saray, S., Tahmoresnezhad, J. et al. Multi-source domain adaptation for image classification. Machine Vision and Applications 31, 44 (2020). https://doi.org/10.1007/s00138-020-01093-2

[2] Griffin, G., Holub, A., and Perona, P. Caltech-256 object category dataset. Technical report, California Institute of Technology, 2007.

[3] Saenko, K., Kulis, B., Fritz, M., and Darrell, T. Adapting visual category models to new domains. In ECCV, 2010.
		
-----------
Contact: Jafar Tahmoresnezhad (tahmores@gmail.com)
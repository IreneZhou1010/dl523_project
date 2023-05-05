# dl523_project
DL523 SP2023
Task: cluster image pixels that contain gene expression data from the cortical brian region into 46 different neuron subtypes and 2 background. 

Dataset: https://drive.google.com/drive/folders/1NThigirSY8VmjtHDLeS8-wdJE9iov0IZ?usp=sharing

generate_mask.py: generate cell masks for files selected

/autoencoder_kmeans: code for implementing approach 2, the autoencoder + k-means pipeline. 
  - simple_kmeans.py: direct k-means clustering of data
  - autoencoder_kmeans_1.py: simple autoencoder with MSE loss
  - CAE_kmeans.py: contractive autoencoder with loss with a penalty term, the squared Frobenius norm of the Jacobian matrix
  - Score_calc.py: calculate score for models and produce figures

UNet folder contains our UNet implementation. 
Attribution: the implementation of U-Net was modified from here https://github.com/milesial/Pytorch-UNet. All other code (training loop, processing methods, etc.) were our own. 

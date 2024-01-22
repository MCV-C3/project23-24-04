# Project
State shortly what you did during each week. Just a table with the main results is enough. Remind to upload a brief presentation (pptx?) at virtual campus. Do not modify the previous weeks code. If you want to reuse it, just copy it on the corespondig week folder.

## Task1 work
In this task we worked on implementing and improving the Bag of Visual Words(BoVW) method. The code in `Task1/BagofVisualWords.ipynb` implements the Bag of Visual Words (BoVW) algorithm for image classification. Helper functions for data loading, creating k folds for k-fold crossvalidation, and metric calculation can be found in `Task1/utils.py`. 

To install the dependencies create a virtual python environment and install the packages from the `requirements.txt` file using:

```pip install -r requirements.txt```


The `Task1/BagofVisualWords.ipynb` notebook includes the following steps:

- Loading and preprocessing the dataset
- Extracting features using SIFT(Scale-Invariant Feature Transform), KAZE and dense-SIFT
- Building a visual vocabulary using K-means clustering with various amounts of clusters
- Encoding images using the Bag of Visual Words representation
- Training and evaluating several classifiers on the encoded features using the Optuna library for parameter optimization
- Attempting to further improve the best performing models using dimensionality reduction and spatial pyramids

The main results of this work include the successful implementation of the BoVW algorithm and the evaluation of its performance on the dataset.

- Best KNN accuracy: 0.830
- Best SVM accuracy: 0.843

The best KNN classification method used 512 clusters for the codebook, k=6 and Manhattan distance for the KNN classifier and 6-component LDA dimensionality reduction
The best SVM classification method used 1024 clusters for the codebook, the linear kernel for the SVM classifier and 64 or 128 components in PCA for dimensionality reduction(both acheved equal accuracy). 


## Task2 work

This week's task was to firstly train an MLP to classify images, and then feed the intermediary activations of the MLP to an SVM or BoVW to classify the image. 
We first defined multiple architectures of MLP-s and tested them with various image sizes and hyperparameters in `gridsearch_neuralnets.py`. After obtaining the best combination of architecture and hyperparameters we extracted the activations of the network at a hidden layer  to obtain feature vectors of length 256 for each image. In `svm.ipynb` and `bow.ipynb` we use gridsearch to obtain the best parameters for either method.

As is visible below none of our our methods developed this managed to outperform those of the previous week, indicating that robust visual descriptors such as SIFT are better at extracting visual features of images for classification than our trained MLP.

| Method/Classifier | Accuracy | Task Number |
|----------|----------|----------|
| BoVW + SVM |  0.843 | 1 |
| BoVw + KNN | 0.830 | 1 |
| MLP + SVM | 0.649 | 2 |
| MLP | 0.633 | 2 |
| MLP + BoVW | 0.592 | 2 |


## Task3 work
| Method/Classifier | Accuracy | Task Number | Dataset |
|----------|----------|----------|----------|
| BoVW + SVM |  0.843 | 1 | MIT_split | 
| BoVw + KNN | 0.830 | 1 | MIT_split | 
| MLP + SVM | 0.649 | 2 | MIT_split | 
| MLP | 0.633 | 2 | MIT_split | 
| MLP + BoVW | 0.592 | 2 | MIT_split | 
| InceptionResnetV2 (Best) | 0.94 | 3 | MIT_split | 
| InceptionResnetV2 (Finetune Backbone) | 0.92 | 3 | MIT_split | 
| InceptionResNetV2 | 0.89 | 3 | MIT_small_1 | 
## Task4 work


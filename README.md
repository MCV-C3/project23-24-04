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

# Steps to get intermediate layer activations #
1. Pull this repo from github
2. Start/activate virtual environment
3. Install python dependencies:
```pip install -r requirements.txt```
4. Go to Task2 directory: `cd Task2`
5. Download weights from `https://drive.google.com/drive/u/1/folders/1EtQ9Bw6TV4kxkJbnfDgVJ5IScc51MeRr` and move to `Task2` directory
6. Change `VAL_DIR` and `MODEL_FNAME` variables to your local values
7. Run `classify.py`, the line `layer_output = model_layer.predict(img)` extracts activations at the 2nd to last layer. 
8. To change what layer the activations are extracted from:
    - Check layer names from `MLP.py` (most are named 'first', 'second', 'third', etc, pick one of those)
    - Change the layer name in the line: `model_layer = Model(inputs=model.input, outputs=model.get_layer('second').output)` to your desired layer.



## Task3 work

## Task4 work


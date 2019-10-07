# Cancer Detection Project (Python)

## Dependencies: 
pandas, sklearn, matplotlib, scipy, seaborn, numpy

## Files:
The jupyternotebook is [here](). 

## 4'D' Data Science Framework: 
I use the 4D framework for my data science projects. It is a simple and reliable method of solving a business problem with data.

# Define: 
The kaggle competition is [here].(https://www.kaggle.com/c/histopathologic-cancer-detection/overview/evaluation)
The goal of this project is to identify metastatic cancer in small image patches taken from a version of the PCam benchmark dataset. 
The evaluation metric is area under the ROC curve, which is a good metric for this classification problem since it measures the probability that a true positive classification does better than (outranks) a negative one and is thus independent of the classification threshold. Since I care mostly about the final class prediction and not fine tuning the threshold, this is a good metric.

# Discover:
## Data (See https://github.com/basveeling/pcam and https://www.kaggle.com/c/histopathologic-cancer-detection/data):
* "(96 x 96px) extracted from histopathologic scans of lymph node sections. Each image is annoted with a binary label indicating presence of metastatic tissue."
* A positive label indicates that the center 32x32px region of a patch contains at least one pixel of tumor tissue. Tumor tissue in the outer region of the patch does not influence the label. 
* 262,144 training examples
* 32,768 each cross-validation and test set
* All have a 50/50 balance between positive and negative examples
* "The original PCam dataset contains duplicate images due to its probabilistic sampling, however, the version presented on Kaggle does not contain duplicates."

# Develop:
## Features:

## Models:

## Results (MSE):

## Feature Importance:


# Deploy:



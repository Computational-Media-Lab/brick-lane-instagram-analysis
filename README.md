# Image & NLP Analysis of Instagram Data

Description: Utilizing NLP and image processing methods to analyze different Instagram posts from Brick Lane, London

## Objective

Develop and implement NLP and image processing methods to analyze Instagram posts and determine if any significant bias arises from the computational methods.

## Overview

Implemented sentiment analysis and topic labeling on Instagram posts' text data (captions and tags) to determine if there's any text analysis bias within these computational methods. After, HOG + SVM and CNN Linear Regression Ensemble model was created to analyze the images of the posts to count how many people were in each photo. The results were compared with actual manually labeled results to determine the accuracy and any biases.

## Methods
- Natural Language Processing
- Image Analysis
- ML
- Statistical Analysis
- Model Evaluation

## Languages & Tools
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- TensorFlow
- OpenCV
- Scikit-Image
- VaderSentiment
- BERTopic

## Key Findings
1. Deep Learning image processing methods produced a F1-Score of 0.91 which shows that computational methods are able to strongly accurately detect people within images.
2. Based on manual review of NLP results, NLP methods can moderately accurately detect sentiment and themes from Instagram post captions and tags.

## Repository Information
1. Code Folder  
  a. eda.ipynb: This code is the exploratory data analysis and the NLP analysis for the instagram captions and tags.  
  b. image-analysis.ipynb: This file contains all the image analysis code including HOG + SVM, CNN, and a Linear Regression Ensemble of the two models.
2. Output & Figures Folder  
  a. instagram-analysis-output.md: Contains performance metrics for the HOG + SVM, CNN, and Linear Regression Ensemble models  
  b. HOG_SVM_heatmap_output.png: Contains a heatmap of classifications for the HOG + SVM model  
  c. CNN_heatmap_output.png: Contains a heatmap of classifications for the CNN model 

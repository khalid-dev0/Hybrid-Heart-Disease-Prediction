# Hybrid Heart Disease Prediction System

## Overview
This repository contains a two-stage machine learning architecture designed to predict heart disease risk using the CDC BRFSS 2015 Health Indicators dataset. The project addresses the critical challenge of severe class imbalance (~9.4% positive cases) and heavy feature overlap in behavioral survey data.

## Two-Stage Architecture
In clinical machine learning, missing an actual patient (False Negative) carries a much higher cost than a false alarm. To balance this, the system relies on a hybrid pipeline:

* **Stage 1 (The Wide Net):** A high-sensitivity Deep Neural Network built with PyTorch. It utilizes moderated class weights in the `BCEWithLogitsLoss` function to maximize Recall, capturing as many true patients as possible.
* **Stage 2 (Confidence-Gated Spatial Inspection):** A `OneClassSVM` model trained exclusively on the spatial distribution of confirmed patients. It inspects positive predictions from Stage 1. If a prediction is spatially anomalous and the neural network's confidence is borderline, the classification is flipped to Healthy to eliminate clear False Positives. High-confidence outliers are flagged for medical review.

## Key Findings: The Data Information Limit
Extensive Exploratory Data Analysis (EDA) and diagnostic pivot tables revealed a severe spatial overlap between False Positives and True Positives, as well as between False Negatives and True Negatives. Both groups share nearly identical behavioral profiles (e.g., similar Age, HighBP, BMI, and GenHlth metrics). 

Because this data relies on general survey responses rather than clinical measurements (such as ECGs or Troponin levels), it inherently lacks the discriminative features required to perfectly isolate true patients. The current performance reflects the Bayes Error Rate limit of this specific feature space.

## System Role
Given the informational ceiling of the dataset, this architecture is positioned as an **Early Triage System**. It prioritizes a high Recall to ensure potential patients are flagged for clinical review, accepting a higher False Alarm rate as a necessary safeguard.

## Future Directions
* **Advanced Loss Functions:** Implementing `Focal Loss` to force the neural network to dynamically penalize hard, overlapping examples during gradient descent.
* **Feature Engineering:** Creating composite risk indicators (e.g., a "Metabolic Syndrome Flag") to construct new dimensions for linear separability.

## Tech Stack
* **Deep Learning:** PyTorch
* **Machine Learning:** Scikit-Learn (One-Class SVM, PCA)
* **Data Processing & EDA:** Pandas, NumPy, SciPy, Seaborn, Matplotlib

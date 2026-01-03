# ICU Mortality Risk Prediction

## Overview

This project implements predictive modeling techniques to estimate **ICU patient mortality risk** using structured clinical data. The goal is to build, evaluate, and compare linear classification models that can identify high-risk patients based on observed features, with an emphasis on sound model selection, regularization, and performance evaluation.

The project was developed in the context of **EECS 445: Introduction to Machine Learning (University of Michigan)** and follows a rigorous experimental pipeline including feature preprocessing, model training, cross-validation, and metric-based evaluation.

---

## Models Used

The project primarily focuses on **linear models for binary classification**, including:

- **Logistic Regression**
  - Used as a probabilistic baseline model for mortality prediction
  - Outputs calibrated probabilities for patient risk
- **Ridge-Regularized Linear Models (L2 Regularization)**
  - Controls overfitting in high-dimensional feature spaces
  - Improves generalization and numerical stability

Regularization strength is selected using **k-fold cross-validation**, and models are evaluated using multiple clinically relevant metrics.

---

## Dataset Description

The dataset consists of ICU patient records with engineered numerical features and a binary mortality label.

- **Input features**: Clinical measurements, vitals, or derived indicators
- **Target variable**:
  - `1` → Patient did not survive
  - `0` → Patient survived

The data is split into:
- Training set
- Validation (via cross-validation)
- Test / held-out set

All preprocessing steps are applied consistently across splits.

---

## Project Structure

```
.
├── data/
│ ├── dataset.csv # Main training dataset
│ ├── heldout.csv # Held-out test set
│ └── imbalanced.csv # Optional imbalanced dataset
├── project.py # Main training and evaluation pipeline
├── helper.py # Utility functions (metrics, data loading, CV)
├── test_output.py # Output format validation
└── README.md
```

---

## Feature Processing

- Features are normalized or standardized when appropriate
- Consistent preprocessing is enforced across training and evaluation
- Regularization is critical due to the dimensionality of the feature space

---

## Model Evaluation

Models are evaluated using **k-fold cross-validation** with stratified splits to preserve class balance.

### Performance Metrics

The following metrics are implemented and reported:

- Accuracy  
- Precision  
- Recall (Sensitivity)  
- Specificity  
- F1-Score  
- AUROC (Area Under the ROC Curve)

AUROC is emphasized due to its robustness under class imbalance and relevance in clinical risk prediction.

---

## Hyperparameter Selection

- Regularization strength is selected using grid search over logarithmic ranges
- Cross-validation is implemented manually (not using sklearn’s built-in CV helpers)
- The best hyperparameters are chosen based on mean validation performance

---

## Handling Class Imbalance

The project includes experiments with:

- Balanced vs. imbalanced datasets
- Adjusted class weights to penalize false negatives more heavily
- Analysis of how weighting impacts sensitivity and specificity

This is especially important in ICU settings where **false negatives can be costly**.

---

## Results & Insights

Key observations include:

- Regularization significantly improves generalization
- Ridge-regularized models are more stable in high-dimensional settings
- AUROC is a more informative metric than accuracy for mortality prediction
- Adjusting class weights can meaningfully improve sensitivity on imbalanced data

---

## Requirements

The project was developed using the following environment:

- Python 3.6+
- NumPy
- Pandas
- scikit-learn
- Matplotlib

All dependencies can be installed via Anaconda.

---

## How to Run

1. Install dependencies
2. Place datasets in the `data/` directory
3. Run the main script:

```bash
python project.py

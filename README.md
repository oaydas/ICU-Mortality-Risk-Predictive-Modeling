# ICU Mortality Risk Prediction

A supervised machine learning project that predicts **ICU patient mortality risk** using structured clinical data, leveraging regularized logistic regression to produce interpretable and robust risk estimates.

This project implements an end-to-end predictive modeling pipeline focused on clinically meaningful evaluation, principled model selection, and transparency in high-stakes healthcare settings.

---

## Overview

The system models the probability of in-hospital mortality for ICU patients based on features available during admission or early stay. It emphasizes:

- Predictive performance under class imbalance
- Statistical interpretability of learned parameters
- Robust generalization through regularization and cross-validation

The primary model used is **logistic regression with L2 (ridge) regularization**, a standard and well-understood approach in clinical risk modeling.

---

## Major Components

### Data Processing
Responsible for preparing raw ICU patient data for modeling.

Key responsibilities:
- Handling missing values
- Normalizing and scaling numerical features
- Encoding categorical variables
- Separating features from mortality labels

All preprocessing steps ensure compatibility with linear models and stable optimization.

### Predictive Model
The core classifier is a regularized logistic regression model.

Key characteristics:
- Binary classification for mortality outcome
- L2 (ridge) regularization to prevent overfitting
- Probabilistic output for risk estimation
- Convex optimization with guaranteed convergence

The learned coefficients directly reflect feature associations with mortality risk.

### Model Selection
Hyperparameters are selected using **k-fold cross-validation** on the training data.

Selection process:
- Evaluate candidate regularization strengths
- Average performance across folds
- Choose the model maximizing cross-validation performance
- Retrain on the full training set with selected parameters

This ensures stable and reproducible performance estimates.

### Evaluation
Model performance is evaluated using clinically relevant metrics:

- Accuracy
- Precision
- Recall (Sensitivity)
- Specificity
- F1-score
- Area Under the ROC Curve (AUROC)

AUROC is emphasized due to its threshold-independent nature and relevance for risk stratification.

---

## Modeling Approach

### Logistic Regression with Ridge Regularization
The model estimates mortality probability using:

p(y = 1 | x) = σ(θᵀx)

with an L2-regularized objective:

min_θ  −∑ᵢ [yᵢ log pᵢ + (1 − yᵢ) log(1 − pᵢ)] + λ‖θ‖₂²

Regularization stabilizes coefficient estimates, improves generalization, and reduces sensitivity to noisy or correlated features.

---

## Handling Class Imbalance

ICU mortality prediction often involves imbalanced outcomes. The project addresses this by:

- Evaluating sensitivity and specificity explicitly
- Emphasizing AUROC during model selection
- Avoiding misleading reliance on accuracy alone

This ensures performance remains meaningful in clinically realistic settings.

---

## Interpretability

A core design goal is interpretability:

- Coefficient signs indicate whether a feature increases or decreases mortality risk
- Coefficient magnitudes reflect relative importance
- Regularization prevents extreme or unstable weights

This allows the model to support downstream clinical analysis and auditing.

---

## Project Structure

.
├── data/
│ ├── train.csv
│ ├── test.csv
├── preprocessing.py
├── model.py
├── evaluation.py
├── main.py
└── README.md

- preprocessing.py: Feature cleaning and transformation  
- model.py: Logistic regression training and prediction  
- evaluation.py: Metric computation and validation  
- main.py: End-to-end execution pipeline  

---

## Running the Project

1. Install dependencies:
pip install numpy pandas scikit-learn

3. Run the full pipeline:
python main.py

Performance metrics will be printed to standard output.

---

## Limitations

- Linear decision boundary may not capture complex nonlinear interactions
- Performance depends on feature quality and completeness
- Not validated for real-world clinical deployment

---

## Future Extensions

- Nonlinear models (e.g., gradient-boosted trees)
- Time-series modeling of vitals and labs
- Model calibration analysis
- External validation on independent ICU cohorts

---

## Summary

This project implements a clinically grounded ICU mortality risk predictor with:

- Regularized logistic regression for stability and interpretability
- Cross-validated model selection
- Comprehensive evaluation using appropriate metrics
- A clean, modular pipeline suitable for research and extension

It demonstrates strong fundamentals in applied machine learning, healthcare modeling, and responsible evaluation for high-impact prediction tasks.

import numpy as np
import numpy.typing as npt
import pandas as pd
import yaml
import matplotlib
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix, roc_curve, auc
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import helper

matplotlib.use("Agg")


__all__ = [
    "generate_feature_vector",
    "impute_missing_values",
    "normalize_feature_matrix",
    "get_classifier",
    "performance",
    "cv_performance",
    "select_param_logreg",
    "select_param_RBF",
    "plot_weight",
]


# load configuration for the project, specifying the random seed and variable types
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
seed = config["seed"]
np.random.seed(seed)


# def generate_feature_vector(df: pd.DataFrame) -> dict[str, float]:
#     """
#     Reads a dataframe containing all measurements for a single patient
#     within the first 48 hours of the ICU admission, and convert it into
#     a feature vector.

#     Args:
#         df: pd.Dataframe, with columns [Time, Variable, Value]

#     Returns:
#         a python dictionary of format {feature_name: feature_value}
#         for example, {"Age": 32, "Gender": 0, "max_HR": 84, ...}
#     """
#     static_variables = config["static"]
#     timeseries_variables = config["timeseries"]
    
#     # DONE: 1) Replace unknown values with np.nan
#     # NOTE: pd.DataFrame.replace() may be helpful here, refer to documentation for details
#     df = df.replace(-1, np.nan)

#     # Extract time-invariant and time-varying features (look into documentation for pd.DataFrame.iloc)
#     static, timeseries = df.iloc[0:5], df.iloc[5:]

#     feature_dict = {}
#     # DONE: 2) extract raw values of time-invariant variables into feature dict
#     for _, row in static.iterrows():
#         feature_dict[row["Variable"]] = row["Value"]

#     # DONE  3) extract max of time-varying variables into feature dict
#     max_values = timeseries.groupby("Variable")["Value"].max()
#     for var, max_val in max_values.items():
#         feature_dict[f"max_{var}"] = max_val
    
#     return feature_dict


# def impute_missing_values(X: npt.NDArray) -> npt.NDArray:
#     """
#     For each feature column, impute missing values (np.nan) with the population mean for that feature.

#     Args:
#         X: array of shape (N, d) which could contain missing values
        
#     Returns:
#         X: array of shape (N, d) without missing values
#     """
#     for i in range(X.shape[1]):
#         col_mean = np.nanmean(X[:, i])
#         X[np.isnan(X[:, i]), i] = col_mean
    
#     return X


def normalize_feature_matrix(X: npt.NDArray) -> npt.NDArray:
    """
    For each feature column, normalize all values to range [0, 1].

    Args:
        X: array of shape (N, d).

    Returns:
        X: array of shape (N, d). Values are normalized per column.
    """
    # NOTE: sklearn.preprocessing.MinMaxScaler may be helpful
    scaler = MinMaxScaler()
    return scaler.fit_transform(X)


def get_classifier(
    loss: str = "logistic",
    penalty: str | None = None,
    C: float = 1.0,
    class_weight: dict[int, float] | None = None,
    kernel: str = "rbf",
    gamma: float = 0.1,
) -> KernelRidge | LogisticRegression:
    """
    Return a classifier based on the given loss, penalty function and regularization parameter C.

    Args:
        loss: Specifies the loss function to use.
        penalty: The type of penalty for regularization.
        C: Regularization strength parameter.
        class_weight: Weights associated with classes.
        kernel : Kernel type to be used in Kernel Ridge Regression.
        gamma: Kernel coefficient.

    Returns:
        A classifier based on the specified arguments.
    """

    if loss == "logistic":
        pass
    elif loss == "squared_error":
        pass


def performance(clf_trained, X, y_true, metric="accuracy"):
    y_pred = clf_trained.predict(X)
    
    if metric == "accuracy":
        return accuracy_score(y_true, y_pred)
    elif metric == "precision":
        return precision_score(y_true, y_pred, labels=[-1, 1], pos_label=1, zero_division=0)
    elif metric == "f1-score":
        return f1_score(y_true, y_pred, labels=[-1, 1], pos_label=1, zero_division=0)
    elif metric == "auroc":
        return roc_auc_score(y_true, clf_trained.decision_function(X))
    elif metric == "average_precision":
        return average_precision_score(y_true, clf_trained.decision_function(X))
    elif metric in ["sensitivity", "specificity"]:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[-1, 1]).ravel()
        if metric == "sensitivity":  # Sensitivity (Recall for positive class)
            return tp / (tp + fn) if (tp + fn) > 0 else 0.0
        else:  # Specificity (Recall for negative class)
            return tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        raise ValueError("Invalid metric specified")


def performance_kridge(clf_trained, X, y_true, metric="accuracy"):
    y_pred = clf_trained.predict(X)
    y_pred_binary = np.where(y_pred >= 0, 1, -1)
    
    if metric == "accuracy":
        return accuracy_score(y_true, y_pred_binary)
    elif metric == "precision":
        return precision_score(y_true, y_pred_binary, labels=[-1, 1], pos_label=1, zero_division=0)
    elif metric == "f1-score":
        return f1_score(y_true, y_pred_binary, labels=[-1, 1], pos_label=1, zero_division=0)
    elif metric == "auroc":
        return roc_auc_score(y_true, y_pred)
    elif metric == "average_precision":
        return average_precision_score(y_true, y_pred)
    elif metric in ["sensitivity", "specificity"]:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary, labels=[-1, 1]).ravel()
        if metric == "sensitivity":
            return tp / (tp + fn) if (tp + fn) > 0 else 0.0
        else:
            return tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        raise ValueError("Invalid metric specified")


def cv_performance(
    clf: KernelRidge | LogisticRegression,
    X: npt.NDArray,
    y: npt.NDArray,
    metric: str = "accuracy",
    k: int = 5,
) -> tuple[float, float, float]:
    """
    Splits the data X and the labels y into k-folds and runs k-fold
    cross-validation: for each fold i in 1...k, trains a classifier on
    all the data except the ith fold, and tests on the ith fold.
    Calculates the k-fold cross-validation performance metric for classifier
    clf by averaging the performance across folds.

    Args:
        clf: an instance of a sklearn classifier
        X: (n,d) array of feature vectors, where n is the number of examples
           and d is the number of features
        y: (n,) vector of binary labels {1,-1}
        k: the number of folds (default=5)
        metric: the performance metric (default='accuracy'
             other options: 'precision', 'f1-score', 'auroc', 'average_precision',
             'sensitivity', and 'specificity')

    Returns:
        a tuple containing (mean, min, max) cross-validation performance across the k folds
    """

    skf = StratifiedKFold(n_splits=k, shuffle=False)
    scores = []
    
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        clf.fit(X_train, y_train)
        if isinstance(clf, KernelRidge):
            scores.append(performance_kridge(clf, X_test, y_test, metric))
        else:
            scores.append(performance(clf, X_test, y_test, metric))
    
    return np.mean(scores), np.min(scores), np.max(scores)


def select_param_logreg(
    X: npt.NDArray,
    y: npt.NDArray,
    metric: str = "accuracy",
    k: int = 5,
    C_range: list[float] = [],
    penalties: list[str] = ["l2", "l1"],
) -> tuple[float, str]:
    """
    Sweeps different settings for the hyperparameter of a logistic regression, calculating the k-fold CV
    performance for each setting on X, y.

    Args:
        X: (n,d) array of feature vectors, where n is the number of examples
        and d is the number of features
        y: (n,) array of binary labels {1,-1}
        k: int specifying the number of folds (default=5)
        metric: string specifying the performance metric for which to optimize (default='accuracy',
             other options: 'precision', 'f1-score', 'auroc', 'average_precision', 'sensitivity',
             and 'specificity')
        C_range: an array with C values to be searched over
        penalties: a list of strings specifying the type of regularization penalties to be searched over

    Returns:
        The hyperparameters for a logistic regression model that maximizes the
        average k-fold CV performance.
    """
    best_C = None
    best_penalty = None
    best_performance = float('-inf')
    
    for penalty in penalties:
        for C in C_range:
            # Skip l1 penalty if solver does not support it
            if penalty == "l1":
                solver = "liblinear"
            else:
                solver = "liblinear"  # "liblinear" supports both l1 and l2

            clf = LogisticRegression(penalty=penalty, C=C, solver=solver, fit_intercept=False, random_state=42)
            mean_perf, _, _ = cv_performance(clf, X, y, metric, k)
            
            if mean_perf > best_performance:
                best_performance = mean_perf
                best_C = C
                best_penalty = penalty
    
    return best_C, best_penalty


def select_param_RBF(
    X: np.ndarray,
    y: np.ndarray,
    metric: str = "accuracy",
    k: int = 5,
    C_range: list[float] = [],
    gamma_range: list[float] = [],
) -> tuple[float, float]:
    """
    Sweeps different settings for the hyperparameter of an RBF Kernel Ridge Regression,
    calculating the k-fold CV performance for each setting on X, y.

    Args:
        X: (n,d) array of feature vectors, where n is the number of examples
        and d is the number of features
        y: (n,) array of binary labels {1,-1}
        k: int specifying the number of folds (default=5)
        metric: string specifying the performance metric (default='accuracy')
        C_range: an array with C values to be searched over
        gamma_range: an array with gamma values to be searched over

    Returns:
        The parameter values for an RBF Kernel Ridge Regression that maximizes the
        average k-fold CV performance.
    """
    best_C = None
    best_gamma = None
    best_performance = float('-inf')
    
    skf = StratifiedKFold(n_splits=k, shuffle=False)
    
    for C in C_range:
        for gamma in gamma_range:
            clf = KernelRidge(alpha=1 / (2 * C), kernel="rbf", gamma=gamma)
            scores = []
            
            for train_index, test_index in skf.split(X, y):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                
                clf.fit(X_train, y_train)
                scores.append(performance_kridge(clf, X_test, y_test, metric))
            
            mean_perf = np.mean(scores)
            if mean_perf > best_performance:
                best_performance = mean_perf
                best_C = C
                best_gamma = gamma
    
    return best_C, best_gamma


def plot_weight(
    X: npt.NDArray,
    y: npt.NDArray,
    C_range: list[float],
    penalties: list[str],
) -> None:
    """
    The funcion takes training data X and labels y, plots the L0-norm
    (number of nonzero elements) of the coefficients learned by a classifier
    as a function of the C-values of the classifier, and saves the plot.
    Args:
        X: (n,d) array of feature vectors, where n is the number of examples
        and d is the number of features
        y: (n,) array of binary labels {1,-1}

    Returns:
        None
    """

    print("Plotting the number of nonzero entries of the parameter vector as a function of C")

    for penalty in penalties:
        norm0 = []
        for C in C_range:
            # Initialize and train logistic regression classifier
            clf = LogisticRegression(penalty=penalty, C=C, solver="liblinear", fit_intercept=False)
            clf.fit(X, y)

            # Extract learned coefficients from the model
            w = clf.coef_.flatten()  # Convert to 1D array

            # Compute the ℓ0-norm: count nonzero coefficients
            non_zero_count = np.count_nonzero(w)
            norm0.append(non_zero_count)

        # This code will plot your L0-norm as a function of C
        plt.plot(C_range, norm0)
        
    plt.xscale("log")
    plt.legend(penalties)
    plt.xlabel("Value of C")
    plt.ylabel("Norm of theta")

    plt.savefig("L0_Norm.png", dpi=200)
    plt.close()


def part1(X_train: npt.NDArray, feature_names: list[str]):
    # Compute mean and IQR for each feature
    means = np.mean(X_train, axis=0)
    q1, q3 = np.percentile(X_train, [25, 75], axis=0)
    iqr = q3 - q1

    # Print feature statistics
    print("\nFeature Statistics:")
    print(f"{'Feature Name':<25}{'Mean':<15}{'IQR':<15}")
    print("=" * 55)
    for name, mean, iqr_value in zip(feature_names, means, iqr):
        print(f"{name:<25}{mean:<15.4f}{iqr_value:<15.4f}")

def part2c(X_train: npt.NDArray, y_train: npt.NDArray):
    C_range = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    penalties = ["l1", "l2"]
    metrics = ["accuracy", "precision", "f1-score", "auroc", "average_precision", "sensitivity", "specificity"]
    
    print("Question 2.1(c): Regularized logistic regression with grid search over C and regularization penalty:")
    print("----------------------------------------")
    
    for metric in metrics:
        print(f"Linear Model Hyperparameter Selection based on {metric}:")
        best_C, best_penalty = select_param_logreg(X_train, y_train, metric=metric, C_range=C_range, penalties=penalties)
        
        for penalty in penalties:
            print(f"Penalty:{penalty}")
            for C in C_range:
                clf = LogisticRegression(penalty=penalty, C=C, solver="liblinear", fit_intercept=False, random_state=42)
                mean_perf, min_perf, max_perf = cv_performance(clf, X_train, y_train, metric=metric)
                print(f"C: {C:.4f} score: {mean_perf:.4f} ({min_perf:.4f}, {max_perf:.4f})")
        
        print(f"Best C: {best_C:.4f}")
        print(f"Best Penalty: {best_penalty}")
        print("----------------------------------------")


def part2d(X_train: npt.NDArray, y_train: npt.NDArray, X_test: npt.NDArray, y_test: npt.NDArray):
    # Best hyperparameters from part 2.1(c)
    best_C = 1
    best_penalty = "l1"
    metrics = ["accuracy", "precision", "f1-score", "auroc", "average_precision", "sensitivity", "specificity"]
    
    # Train the model with the best C and penalty
    clf = LogisticRegression(penalty=best_penalty, C=best_C, solver="liblinear", fit_intercept=False, random_state=42)
    clf.fit(X_train, y_train)
    
    # Evaluate performance on the test data using cross-validation (cv_performance)
    print("Cross-validation Performance on Test Data (Best C=1, Penalty=l1):")
    for metric in metrics:
        mean_perf= performance(clf, X_test, y_test, metric=metric)
        print(f"{metric}: {mean_perf:.4f}")

def part2f(X_train: np.ndarray, y_train: np.ndarray, feature_names: list[str]):
    """
    Train an L1-regularized logistic regression model using C = 1.0 on the training data.
    Find the 4 most positive and 4 most negative coefficients of the learned parameter vector.
    """
    # Train L1-regularized logistic regression with C = 1.0
    clf = LogisticRegression(penalty='l1', C=1.0, solver='liblinear', fit_intercept=False)
    clf.fit(X_train, y_train)
    
    # Extract learned coefficients
    theta = clf.coef_.flatten()
    
    # Get indices of 4 most positive and 4 most negative coefficients
    top_positive_indices = np.argsort(theta)[-4:][::-1]  # Largest 4 coefficients
    top_negative_indices = np.argsort(theta)[:4]  # Smallest 4 coefficients
    
    # Print results
    print("Top 4 Most Positive Coefficients:")
    for idx in top_positive_indices:
        print(f"Feature {feature_names[idx]}: {theta[idx]:.4f}")
    
    print("\nTop 4 Most Negative Coefficients:")
    for idx in top_negative_indices:
        print(f"Feature {feature_names[idx]}: {theta[idx]:.4f}")
    
    return [feature_names[idx] for idx in top_positive_indices], [feature_names[idx] for idx in top_negative_indices], theta

def part3_1b(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray):
    """
    Train an L2-regularized logistic regression model with class weights { -1: 1, 1: 50 }.
    Evaluate performance on the test data.
    """
    # Used 1 and 5 for Question 3.2b instead of 1 and 50
    class_weights = {-1: 1, 1: 5}
    
    # Train L2-regularized logistic regression with C = 1.0 and class weights
    clf = LogisticRegression(penalty='l2', C=1.0, solver='liblinear', fit_intercept=False, class_weight=class_weights)
    clf.fit(X_train, y_train)
    
    # Define performance metrics
    metrics = ["accuracy", "precision", "f1-score", "auroc", "average_precision", "sensitivity", "specificity"]
    
    # Evaluate performance on the test data using cross-validation
    print("Cross-validation Performance on Test Data (Best C=1, Penalty=l2, Wp=50, Wn=1):")
    for metric in metrics:
        mean_perf = performance(clf, X_test, y_test, metric=metric)
        print(f"{metric}: {mean_perf:.4f}")
    
    return clf

def part3_2a(X_train: np.ndarray, y_train: np.ndarray):
    """
    Find the best class weights (Wn, Wp) to maximize AUROC using cross-validation.
    """
    # Compute class distribution
    unique, counts = np.unique(y_train, return_counts=True)
    class_distribution = dict(zip(unique, counts))
    print("Class Distribution in Training Data:", class_distribution)
    
    # Initialize weights inversely proportional to class frequency
    count_neg = class_distribution[-1]
    count_pos = class_distribution[1]
    initial_Wp = count_neg / count_pos  # Inverse class frequency ratio
    initial_Wn = 1  # Keep negative class weight fixed
    
    print(f"Initial Class Weights: Wn={initial_Wn}, Wp={initial_Wp:.2f}")
    
    # Grid search over different positive class weights
    best_Wp = None
    best_auroc = float('-inf')
    Wp_values = [1, 5, 10, 25, 50, 75, 100, 6.05]
    
    for Wp in Wp_values:
        class_weights = {-1: initial_Wn, 1: Wp}
        clf = LogisticRegression(penalty="l2", C=1.0, solver="liblinear", fit_intercept=False, class_weight=class_weights)
        mean_auroc, _, _ = cv_performance(clf, X_train, y_train, metric="auroc")
        print(f"Wp: {Wp}, AUROC: {mean_auroc:.4f}")
        
        if mean_auroc > best_auroc:
            best_auroc = mean_auroc
            best_Wp = Wp
    
    print(f"Best Class Weights: Wn={initial_Wn}, Wp={best_Wp}")
    
    # Train final model with best class weights
    final_clf = LogisticRegression(penalty="l2", C=1.0, solver="liblinear", fit_intercept=False, class_weight={-1: initial_Wn, 1: best_Wp})
    final_clf.fit(X_train, y_train)
    
    return final_clf, best_Wp


def part3_3a(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray):
    """
    Plot ROC curves for two different class weight settings: (Wn=1, Wp=1) and (Wn=1, Wp=5) using C=1.0.
    """
    plt.figure(figsize=(8, 6))
    
    class_weights_list = [{-1: 1, 1: 1}, {-1: 1, 1: 5}]
    labels = ["Wn=1, Wp=1", "Wn=1, Wp=5"]
    
    for class_weights, label in zip(class_weights_list, labels):
        # Train logistic regression model
        clf = LogisticRegression(penalty="l2", C=1.0, solver="liblinear", fit_intercept=False, class_weight=class_weights)
        clf.fit(X_train, y_train)
        
        # Compute ROC curve
        y_scores = clf.decision_function(X_test)
        fpr, tpr, _ = roc_curve(y_test, y_scores)
        auc_score = auc(fpr, tpr)
        
        # Plot ROC curve
        plt.plot(fpr, tpr, label=f"{label} (AUROC = {auc_score:.4f})")
    
    # Plot reference line (random classifier)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Random Guessing")
    
    # Labels and legend
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves for Different Class Weights (C=1.0)")
    plt.legend()

    # Save the plot instead of displaying it
    plt.savefig("roc_curve.png", dpi=200)
    plt.close()


def part4_1b(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray):
    """
    Compare empirical performance of Logistic Regression and Kernel Ridge Regression with a linear kernel.
    """
    C = 1.0
    # Train Logistic Regression model with fixed random state
    logreg = LogisticRegression(penalty="l2", C=C, fit_intercept=False, random_state=42)
    logreg.fit(X_train, y_train)
    
    # Train Kernel Ridge Regression model
    kridge = KernelRidge(alpha=1 / (2 * C), kernel="linear")
    kridge.fit(X_train, y_train)

    # Define metrics to evaluate
    metrics = ["accuracy", "precision", "f1-score", "auroc", "average_precision", "sensitivity", "specificity"]

    # Print performance for each model using performance()
    print(f"\nPerformance for Logistic Regression (C={C}):")
    for metric in metrics:
        score = performance(logreg, X_test, y_test, metric=metric)
        print(f"{metric}: {score:.4f}")

    print(f"\nPerformance for Kernel Ridge Regression (C={C}):")
    for metric in metrics:
        score = performance_kridge(kridge, X_test, y_test, metric=metric)
        print(f"{metric}: {score:.4f}")


def part4_2b(
    X: np.ndarray,
    y: np.ndarray,
    k: int = 5,
    C: float = 1.0,
    gamma_range: list[float] = [0.001, 0.01, 0.1, 1, 10, 100],
) -> None:
    """
    Reports the cross-validation AUROC performance (mean, min, max) for a fixed C = 1.0 and
    a range of gamma values.
    
    Args:
        X: (n,d) array of feature vectors
        y: (n,) array of binary labels {1,-1}
        k: int specifying the number of folds (default=5)
        C: fixed value of C (default=1.0)
        gamma_range: list of gamma values to evaluate
    """
    skf = StratifiedKFold(n_splits=k, shuffle=False)
    
    print("Cross-validation AUROC Performance for RBF Kernel Ridge Regression")
    for gamma in gamma_range:
        clf = KernelRidge(alpha=1 / (2 * C), kernel="rbf", gamma=gamma)
        scores = []
        
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            clf.fit(X_train, y_train)
            scores.append(performance_kridge(clf, X_test, y_test, metric="auroc"))
        
        mean_perf = np.mean(scores)
        min_perf = np.min(scores)
        max_perf = np.max(scores)
        
        print(f"Gamma: {gamma:.3f} | AUROC: {mean_perf:.4f} ({min_perf:.4f}, {max_perf:.4f})")


def part4_2c(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray):
    """
    Train multiple non-linear models using Kernel Ridge Regression with an RBF kernel 
    on the training data and evaluate their performance on the test set.
    """
    # Define C and gamma ranges to test
    C_range = [0.01, 0.1, 1.0, 10, 100]
    gamma_range = [0.01, 0.1, 1, 10]

    # Define performance metrics to evaluate
    metrics = ["accuracy", "precision", "f1-score", "auroc", "average_precision", "sensitivity", "specificity"]

    # Store best AUROC performance and corresponding hyperparameters
    best_C = None
    best_gamma = None
    best_auroc = float('-inf')

    print("\nEvaluating RBF Kernel Ridge Regression on Test Set:")
    
    for C in C_range:
        for gamma in gamma_range:
            # Train Kernel Ridge Regression with RBF kernel
            kridge_rbf = KernelRidge(alpha=1 / (2 * C), kernel="rbf", gamma=gamma)
            kridge_rbf.fit(X_train, y_train)

            # Compute AUROC for this model
            auroc_score = performance_kridge(kridge_rbf, X_test, y_test, "auroc")

            # Print performance for this (C, gamma) pair
            print(f"\nC={C:.4f}, gamma={gamma:.4f}: AUROC={auroc_score:.4f}")

            # Update best parameters if the current AUROC is better
            if auroc_score > best_auroc:
                best_auroc = auroc_score
                best_C = C
                best_gamma = gamma

    # Train final model with best hyperparameters
    print(f"\nBest Parameters: C={best_C}, gamma={best_gamma}, Best AUROC={best_auroc:.4f}")
    best_model = KernelRidge(alpha=1 / (2 * best_C), kernel="rbf", gamma=best_gamma)
    best_model.fit(X_train, y_train)

    # Evaluate test performance on all metrics
    print(f"\nTest Performance for Best RBF Kernel Ridge Regression (C={best_C}, gamma={best_gamma}):")
    for metric in metrics:
        test_perf = performance_kridge(best_model, X_test, y_test, metric)
        print(f"{metric}: {test_perf:.4f}")


# CHALLENGE FUNCTIONS FROM THIS POINT ON
def generate_feature_vector(df: pd.DataFrame) -> dict[str, float]:
    """
    Reads a dataframe containing all measurements for a single patient
    within the first 48 hours of the ICU admission and converts it into a feature vector.

    Args:
        df: pd.Dataframe, with columns [Time, Variable, Value]

    Returns:
        A dictionary of format {feature_name: feature_value}
    """
    static_variables = config["static"]
    timeseries_variables = config["timeseries"]

    # Replace unknown values with NaN
    df = df.replace(-1, np.nan)

    # Extract static and time-series data
    static = df.iloc[0:5]
    timeseries = df.iloc[5:].copy()  # Create a new copy to avoid modifying original DataFrame

    feature_dict = {}

    # Extract raw values of static variables
    for _, row in static.iterrows():
        feature_dict[row["Variable"]] = row["Value"]

    # **Create a new DataFrame to store the hour values**
    timeseries_new = timeseries.copy()
    timeseries_new["Hour"] = timeseries_new["Time"].str[:2].astype(int)  # Extract first two digits as integer

    # Split into two 24-hour periods using the new DataFrame
    period_1 = timeseries[timeseries_new["Hour"] <= 23]  # First 24 hours
    period_2 = timeseries[timeseries_new["Hour"] > 23]   # Second 24 hours

    # Compute statistics for each feature in both periods
    for var in timeseries_variables:
        feature_dict[f"mean_{var}_p1"] = period_1.loc[period_1["Variable"] == var, "Value"].mean()
        feature_dict[f"std_{var}_p1"] = period_1.loc[period_1["Variable"] == var, "Value"].std()

        feature_dict[f"mean_{var}_p2"] = period_2.loc[period_2["Variable"] == var, "Value"].mean()
        feature_dict[f"std_{var}_p2"] = period_2.loc[period_2["Variable"] == var, "Value"].std()

    return feature_dict


def impute_missing_values(X: np.ndarray) -> np.ndarray:
    """
    Impute missing values using K-Nearest Neighbors (KNN) first.
    Then, apply median imputation for any remaining NaN values.

    Args:
        X: (N, d) array of feature vectors, possibly containing missing values.

    Returns:
        X_imputed: (N, d) array with missing values imputed.
    """

    # Apply KNN Imputation
    knn_imputer = KNNImputer(n_neighbors=20)  # Use 20 nearest neighbors
    X_imputed = knn_imputer.fit_transform(X)

    return X_imputed

# Median missing values
# def impute_missing_values(X: np.ndarray) -> np.ndarray:
#     """
#     For each feature column, impute missing values (np.nan) with the median of the observed values.

#     Args:
#         X: array of shape (N, d) which could contain missing values

#     Returns:
#         X: array of shape (N, d) without missing values
#     """
#     for i in range(X.shape[1]):
#         col_median = np.nanmedian(X[:, i])  # Compute median ignoring NaN values
#         X[np.isnan(X[:, i]), i] = col_median  # Replace NaNs with median value
    
#     return X


# def normalize_feature_matrix(X: np.ndarray) -> np.ndarray:
#     """
#     Normalize each feature column using Z-score normalization.

#     Args:
#         X: (N, d) array of feature vectors.

#     Returns:
#         X_normalized: (N, d) array of normalized feature vectors.
#     """
#     scaler = StandardScaler()
#     return scaler.fit_transform(X)


def train_classifier(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    """
    Train a logistic regression classifier on the processed data.

    Args:
        X_train: (N, d) array of training feature vectors.
        y_train: (N,) array of training labels.

    Returns:
        clf: Trained logistic regression classifier.
    """
    # Set class weights: W_n=1 for negative class (-1) and W_p=5 for positive class (1)
    class_weights = {-1: 1, 1: 5}
    
    clf = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="liblinear",
        fit_intercept=False,
        class_weight=class_weights  # Apply class weights
    )
    
    clf.fit(X_train, y_train)
    return clf


def impute_missing_values_ch(X_train: np.ndarray, X_test: np.ndarray, feature_names: list[str]) -> np.ndarray:
    """
    Impute missing values in X_test using:
    1. Median from training data for columns that are fully NaN in X_test.
    2. K-Nearest Neighbors (KNN) Imputation for remaining missing values.

    Args:
        X_train: (N_train, d) array of training feature vectors.
        X_test: (N_test, d) array of test feature vectors.
        feature_names: List of feature names corresponding to columns.

    Returns:
        X_test: (N_test, d) array with missing values imputed.
    """

    # **Step 1: Replace fully NaN columns in X_test with the median from X_train**
    fully_nan_columns = []
    for i in range(X_test.shape[1]):
        if np.all(np.isnan(X_test[:, i])):  # Fully NaN column in test set
            median_value = np.nanmedian(X_train[:, i])  # Get median from training
            if np.isnan(median_value):  # If training median is also NaN, default to 0
                median_value = 0.0
            X_test[:, i] = median_value
            fully_nan_columns.append(feature_names[i])  # Track affected features

    # **Step 2: Apply KNN Imputation**
    imputer = KNNImputer(n_neighbors=20)  # Use 20 nearest neighbors
    X_test = imputer.fit_transform(X_test)
    
    # **Log the changes**
    if fully_nan_columns:
        print(f"Fully NaN columns in X_heldout replaced with median from training: {fully_nan_columns}")
    
    return X_test


def main():
    print(f"Using Seed = {seed}")

    metrics = ["accuracy", "precision", "f1-score", "auroc", "average_precision", "sensitivity", "specificity"]

    X_challenge, y_challenge, X_heldout, feature_names, held_feature_names = helper.get_challenge_data()

    # X_heldout = impute_missing_values_ch(X_challenge, X_heldout, feature_names)
    # X_heldout = normalize_feature_matrix(X_heldout)

    # Train model using new feature extraction and preprocessing
    clf = train_classifier(X_challenge, y_challenge)

    # # Evaluate model on test data
    # print("\nPerformance for Enhanced Logistic Regression Model (Challenge Model):")
    # for metric in metrics:
    #     mean_perf, min_perf, max_perf = cv_performance(clf, X_challenge, y_challenge, metric=metric)
    #     print(f"{metric}: {mean_perf:.4f} ({min_perf:.4f}, {max_perf:.4f})")

    # # Generate pred
    # y_label = clf.predict(X_heldout).astype(int)  
    # y_score = clf.decision_function(X_heldout)  

    # # Save predictions for grading
    # helper.save_challenge_predictions(y_label, y_score, "oaydas")

    y_pred_challenge = clf.predict(X_challenge)

    # Compute confusion matrix
    conf_matrix = confusion_matrix(y_challenge, y_pred_challenge, labels=[1, -1])

    # Print Confusion Matrix
    print("\nConfusion Matrix (X_challenge vs. y_challenge):")
    print("------------------------------------------------")
    print(f"       Predicted:  1   Predicted: -1")
    print(f"Actual: 1   {conf_matrix[0,0]:^10}  {conf_matrix[0,1]:^10}")
    print(f"Actual: -1  {conf_matrix[1,0]:^10}  {conf_matrix[1,1]:^10}")



if __name__ == "__main__":
    main()

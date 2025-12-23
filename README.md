# Multimodal Machine Learning Framework for EEG-Based Stress Classification

## Overview
This repository presents a comprehensive, research-oriented machine learning framework for **binary stress state classification** using EEG-derived physiological signals. The work emphasizes rigorous data preprocessing, class balancing, exploratory statistical analysis, comparative modeling, and systematic hyperparameter optimization. The pipeline is designed to reflect experimental discipline, reproducibility, and analytical depth expected in doctoral-level research and empirical machine learning studies.

The project investigates how classical machine learning models perform on carefully curated EEG features for distinguishing **Stressed** versus **Unstressed** emotional states, supported by extensive validation and diagnostic analysis.

---

## Dataset Description and Preprocessing
The dataset consists of multichannel EEG measurements collected across multiple subjects and time instances. Initial preprocessing includes:
- Removal of non-informative metadata (timestamps and subject identifiers)
- Consolidation of emotional labels into a binary stress taxonomy
- Detection and handling of duplicate records
- Label encoding of emotional states
- Feature standardization using z-score normalization

To mitigate class imbalance and ensure statistical fairness, controlled subsampling and recombination strategies are applied to construct a balanced dataset prior to training.

---

## Exploratory Data Analysis
Extensive exploratory analysis is performed to characterize the dataset:
- Class distribution visualization to confirm balance
- Pearson correlation analysis across EEG channels
- Distributional analysis of individual EEG electrodes using kernel density estimation
- Heatmap-based visualization of inter-feature dependencies

These steps provide insight into signal relationships and guide model selection.

---

## Experimental Protocol
The dataset is partitioned into:
- Training set
- Validation set
- Test set

Stratified and randomized splitting ensures robustness and prevents information leakage. Performance is assessed exclusively on held-out test data.

---

## Models Implemented
A diverse set of supervised learning models is implemented and evaluated:

- Support Vector Machine (RBF kernel)
- k-Nearest Neighbors
- Random Forest Classifier
- Decision Tree Classifier

Each model is trained using a pseudo-epoch batching strategy to analyze learning dynamics typically associated with iterative optimization, despite the non-gradient-based nature of some algorithms.

---

## Training Dynamics and Evaluation
For each classifier, the following metrics are tracked:
- Training accuracy per epoch
- Validation accuracy per epoch
- Training log-loss
- Validation log-loss

Performance trends are visualized to analyze convergence behavior, stability, and generalization characteristics.

Final evaluation includes:
- Test-set accuracy
- Confusion matrix visualization
- Precision, recall, and F1-score reporting

---

## Model Interpretability
The framework incorporates interpretability mechanisms, including:
- Feature correlation analysis
- Visualization of learned decision structures for decision trees
- Confusion matrix heatmaps for error pattern analysis

These tools enable qualitative inspection of classifier behavior beyond aggregate metrics.

---

## Hyperparameter Optimization
To enhance model performance and robustness, advanced hyperparameter search strategies are applied:

### Random Forest Optimization
- Randomized search over tree depth, number of estimators, feature selection strategies, and sampling configurations
- Quantitative comparison between baseline and optimized models
- Explicit computation of performance improvement

### Support Vector Machine Optimization
- Randomized hyperparameter search over kernel types, regularization strength, and kernel-specific parameters
- Cross-validated selection of optimal configurations
- Final evaluation on unseen test data

---

## Key Contributions
- End-to-end EEG-based stress classification pipeline
- Balanced dataset construction with rigorous preprocessing
- Comparative evaluation of multiple classical ML models
- Learning-curve-style analysis for non-neural classifiers
- Integrated hyperparameter optimization and performance benchmarking
- Emphasis on reproducibility, interpretability, and experimental rigor

---

## Technologies Used
Python, NumPy, Pandas, Matplotlib, Seaborn, scikit-learn

---

## Research Significance
This work demonstrates how carefully engineered classical machine learning systems, when combined with principled data handling and evaluation protocols, can yield strong baselines for physiological signal classification. The framework is suitable for extension toward deep learning, multimodal fusion, or subject-adaptive modeling in future research.

---

## Author
This repository reflects a research-driven implementation intended for advanced academic portfolios, emphasizing methodological clarity, analytical depth, and reproducibility in applied machine learning.

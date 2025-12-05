# Deep Learning Midterm Project

**Author:** Sahrul Ridho Firdaus - 1103223009

## 📋 Purpose

This repository contains the midterm project for the Deep Learning course, demonstrating the application of machine learning and deep learning techniques across three different problem domains: classification, regression, and clustering.

## 📁 Project Overview

The repository consists of three Jupyter notebooks, each addressing a distinct machine learning task:

| Notebook | Task | Dataset Domain |
|----------|------|----------------|
| `midterm-DL-1.ipynb` | Binary Classification | Fraud Transaction Detection |
| `midterm-DL-2 regresi.ipynb` | Regression | Numerical Prediction |
| `midterm_DL_3_clustering.ipynb` | Clustering | Customer Segmentation |

---

## 📓 Notebook Descriptions

### 1. Fraud Transaction Detection (`midterm-DL-1.ipynb`)

**Objective:** Build an end-to-end fraud detection system comparing deep learning and traditional machine learning approaches.

**Pipeline:**
- Data preprocessing with feature engineering (log transforms, time-based features, card aggregations)
- Class imbalance handling using SMOTE (Synthetic Minority Over-sampling Technique)
- Label encoding for categorical variables
- Feature scaling with StandardScaler

**Models Implemented:**
| Model | Architecture/Configuration |
|-------|---------------------------|
| **PyTorch MLP** | 3 hidden layers (256→128→64), BatchNorm, Dropout, ReLU activation |
| **LightGBM** | Gradient boosting with `is_unbalance=True`, 64 leaves, max depth 8 |

**Training Techniques:**
- AdamW optimizer with weight decay
- CosineAnnealingLR scheduler
- Early stopping based on validation AUC

**Evaluation Metrics:**
- ROC-AUC Score (primary metric)
- Automatic best model selection based on validation performance

---

### 2. Regression Model (`midterm-DL-2 regresi.ipynb`)

**Objective:** Predict a continuous target variable using a deep learning regression model.

**Pipeline:**
- Data cleaning and duplicate removal
- Feature engineering pipeline with:
  - Correlation-based feature filtering (threshold: 0.95)
  - Optional PCA for dimensionality reduction
  - Optional polynomial feature generation
- Train/Validation/Test split (64%/16%/20%)
- Target variable scaling with StandardScaler

**Model Architecture:**
| Component | Configuration |
|-----------|---------------|
| **Type** | Multi-Layer Perceptron (MLP) |
| **Layers** | 4 hidden layers (512→256→128→64) |
| **Activation** | ReLU |
| **Regularization** | BatchNorm + Dropout (5%) |
| **Optimizer** | AdamW (lr=0.0003, weight_decay=1e-5) |
| **Scheduler** | CosineAnnealingWarmRestarts |

**Evaluation Metrics:**
| Metric | Description |
|--------|-------------|
| MAE | Mean Absolute Error |
| RMSE | Root Mean Squared Error |
| R² | Coefficient of Determination |

**Results:** Model achieves approximately R² ≈ 0.40 on the test set.

---

### 3. Customer Clustering (`midterm_DL_3_clustering.ipynb`)

**Objective:** Segment customers based on spending and payment behavior using unsupervised learning with deep learning enhancement.

**Pipeline:**
- Drop irrelevant columns (CUST_ID)
- Handle missing values with median imputation
- Feature scaling with StandardScaler

**Models Implemented:**
| Model | Description |
|-------|-------------|
| **Autoencoder** | Deep neural network for dimensionality reduction (64→32→10→32→64) |
| **K-Means** | Clustering on latent representations |
| **DEC (Deep Embedded Clustering)** | End-to-end deep clustering with soft assignment |

**Techniques:**
- Elbow Method for optimal cluster selection
- Silhouette Score for cluster quality evaluation
- t-SNE for 2D visualization of clusters
- Student's t-distribution for soft cluster assignment

**Evaluation:**
- Silhouette Score across different k values (2-10)
- Visual inspection via t-SNE projection
- Cluster profiling with feature mean analysis

---

## 🛠️ Technologies & Libraries

```
- Python 3.x
- PyTorch / TensorFlow
- scikit-learn
- LightGBM
- imbalanced-learn (SMOTE)
- pandas, numpy
- matplotlib, seaborn
- Optuna (hyperparameter tuning)
```

## 🚀 How to Navigate

1. **Start with Classification:** Open `midterm-DL-1.ipynb` for a complete classification pipeline with model comparison
2. **Explore Regression:** Open `midterm-DL-2 regresi.ipynb` for regression task with MLP and feature engineering
3. **Learn Clustering:** Open `midterm_DL_3_clustering.ipynb` for unsupervised learning with deep embedded clustering

Each notebook is self-contained and includes:
- Data loading and exploration
- Preprocessing steps
- Model training with progress tracking
- Evaluation and visualization
- Conclusions

## 📊 Results Summary

| Task | Best Model | Primary Metric | Score |
|------|------------|----------------|-------|
| Classification | PyTorch MLP / LightGBM | AUC-ROC | ~0.90+ |
| Regression | MLP (4 layers) | R² Score | ~0.40 |
| Clustering | DEC + K-Means | Silhouette | Varies by k |

---

## 📝 Notes

- All notebooks are designed to run on Google Colab with GPU support
- CUDA availability is checked at the beginning of each notebook
- Random seeds are set for reproducibility

# Iris Flower Classification

A complete **machine learning** project to classify Iris flowers into three species (**Setosa**, **Versicolor**, **Virginica**) based on sepal and petal measurements.

## 📋 Project Overview

- **Dataset**: Classic Iris dataset from scikit-learn (150 samples, 4 numeric features, 3 balanced classes)
- **Objective**: Build and compare multiple classification models to predict the species of Iris flowers
- **Models Implemented**: Logistic Regression, K-Nearest Neighbors (KNN), Decision Tree, and Random Forest
- **Best Performance**: Achieved **0.97 – 1.00** accuracy
- **Key Skills Demonstrated**: Exploratory Data Analysis (EDA), Data Preprocessing, Model Training, Hyperparameter Tuning, Model Evaluation, and Reproducible Code

## 🛠️ Project Structure

```bash
iris-flower-classification/
├── notebooks/              # Jupyter notebooks for EDA
│   └── 01_iris_eda.ipynb
├── src/                    # Core Python modules
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── model_training.py
│   └── evaluation.py
├── models/                 # Trained model files (.pkl)
├── data/                   # (Optional) dataset files
├── main.py                 # Main script to run the full pipeline
├── requirements.txt
├── README.md
└── .gitignore

## 🌐 Interactive Web App

You can run the interactive version using Streamlit:

```bash
streamlit run app.py

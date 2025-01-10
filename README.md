# AI-Project: Predicting Student Performance

Welcome to the **AI-Project**, a comprehensive exploration of predicting student academic performance using machine learning and deep learning techniques. This repository focuses on utilizing the **UCI Student Performance dataset** to analyze various factors influencing students' final grades.

---

## 📜 Project Overview

Predicting student performance is a crucial task in educational research. This project aims to understand the impact of various factors on academic success and provide actionable insights to enhance learning outcomes. The study involves:

- **Dataset**: UCI Student Performance dataset.
- **Objective**: Predict final grades (G3) under three scenarios:
  1. Using all features.
  2. Excluding the second-period grade (G2).
  3. Excluding both the first-period grade (G1) and G2.
- **Prediction Tasks**:
  - **Binary Classification**: Pass/Fail prediction.
  - **5-Level Classification**: Categorizing grades into 5 levels.
  - **Regression**: Predicting continuous final grades.

---

## 🧰 Features

- **Data Preprocessing**:
  - Handling missing values.
  - Encoding categorical variables.
  - Normalizing continuous variables.

- **Machine Learning Models**:
  - Logistic Regression
  - Naive Bayes
  - Decision Trees
  - Random Forest
  - Support Vector Machines (SVM)
  - Neural Networks
  - Deep Learning architectures

- **Evaluation Metrics**:
  - **Classification**: Accuracy, Precision, Recall, F1-score.
  - **Regression**: Root Mean Squared Error (RMSE).

- **Cross-Validation**:
  - Robust evaluation by splitting datasets into training and testing sets.

---

## 📂 Repository Structure

```plaintext
AI-Project/
├── 5-Level-Classification/  # Code and results for 5-level classification tasks
├── Binary-Classification/   # Code and results for binary classification tasks
├── Regression/              # Code and results for regression tasks
├── student-por.csv          # Dataset used for analysis
└── README.md                # Project documentation

APS Failure Detection System – Machine Learning Project

A complete end-to-end Machine Learning pipeline designed to predict Air Pressure System (APS) failures in Scania trucks using high-dimensional and highly imbalanced sensor data.
The goal is to minimize operational cost by identifying potential failures early, using cost-sensitive modeling and real-world deployment.

Live deployed web app URL - https://huggingface.co/spaces/Faizan9-ai/APS-Failure-Prediction-System


 Project Overview

This project analyzes over 170 APS sensor readings from Scania trucks to detect failures in the Air Pressure System. The dataset is highly imbalanced, with only ~1.6% positive failure cases. Since missing a failure (False Negative) is 50× more costly than a false alarm (False Positive), we use a custom APS cost function to select the best model.

The system includes:

Full data cleaning & preprocessing

EDA with detailed visualizations

Handling missing values & feature engineering

Imbalanced learning techniques (class weights, scale_pos_weight)

Model comparison across Logistic Regression, Random Forest, Gradient Boosting, XGBoost

Custom cost metric (APS cost) from the original Scania challenge

Saving the best model & deploying it on Hugging Face using a Gradio web app

Final predictions for any new APS sensor CSV file

 Dataset

Dataset name: APS Failure at Scania Trucks
Source: Kaggle / UCI Machine Learning Repository
Rows: ~60,000
Features: 170+ numerical sensor readings
Target:

neg → Normal truck (encoded as 0)

pos → APS failure (encoded as 1)

Key Challenges:

Strong class imbalance (59000:1000)

High missingness in several features

High-dimensional data

Extreme cost difference between FN (500) and FP (10)

🔧 Tech Stack

Python 3.10+

NumPy, Pandas

Scikit-learn (1.5.1)

XGBoost, RandomForest, LogisticRegression[best model], Gradient boosting

Matplotlib, Seaborn

Gradio (for web deployment)

Hugging Face Spaces (for model hosting)

Joblib (for model serialization)

 Data Preprocessing
✔ Replace "na" with NaN

The dataset uses "na" instead of empty values, so text must be converted to proper numeric missing values.

✔ Convert all sensor columns to numeric

Required for model training.

✔ Drop features with >80% missing

Reduces noise and improves stability.

✔ Impute missing values

Using median imputation inside a scikit-learn pipeline.

✔ Feature Engineering

Created a new feature:

missing_count = total number of missing features per row


This helped models differentiate between sparse & dense sensor patterns.

 Exploratory Data Analysis (EDA)

The following analyses were performed:

Missing-value heatmaps

Feature-wise missing trend analysis

Distribution plots for top correlated features

KDE plots for failure vs. normal class

Correlation matrix & top predictive sensors

Class imbalance analysis

Outlier detection via boxplots

Findings show strong separation in certain key sensors between normal and failure states.

 Modeling

Four ML models were trained using a unified preprocessing pipeline:

Model	Recall	ROC-AUC	APS Cost
Logistic Regression	0.915	0.9647	11760 (best)
XGBoost	0.86	0.9948	14490
Gradient Boosting	0.69	0.988	31390
Random Forest	0.595	0.993	40650
✔ Best Model: Logistic Regression

Due to:

Highest recall

Lowest APS cost

Fewer expensive false negatives

The APS cost is the most important metric in this business problem.

 APS Cost Function (Business Metric)

The cost is computed using the official Scania formula:

False Negative (missed failure) = 500 units
False Positive (false alarm)    = 10 units


So the custom cost function is:

cost = FN * 500 + FP * 10


This was used to select the best model.

 Model Saving

The final best pipeline (preprocessing + model) is saved using:

joblib.dump(best_pipeline, "aps_best_model.pkl")

 Deployment on Hugging Face

An interactive Gradio web application is deployed on Hugging Face Spaces.

Features:

Upload any APS sensor CSV file

Automatic preprocessing and feature engineering

Predict failure probability + class

Uses the trained scikit-learn pipeline

Files in Deployment:

app.py (Gradio interface)

requirements.txt

aps_best_model.pkl

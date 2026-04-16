# Stroke Risk Prediction Using Machine Learning

This project was completed as part of the HarvardX Data Science Professional Certificate. The objective is to build a supervised machine learning model that predicts the likelihood of stroke in patients using clinical and demographic data.

## Project Overview

Stroke is one of the leading causes of death and long-term disability worldwide. Early identification of high-risk patients can significantly reduce mortality and improve outcomes.

In this project, I built a binary classification pipeline to predict stroke risk using two models:
- Logistic Regression as a baseline
- Random Forest as an advanced model

The models were evaluated using AUC-ROC, with a focus on sensitivity due to the medical nature of the problem.

According to the report, the dataset is highly imbalanced with only about 5 percent positive cases, which required careful handling during training :contentReference[oaicite:0]{index=0}.

## Dataset

The dataset used is the Stroke Prediction Dataset from Kaggle.

It contains:
- 5,110 patient records
- 12 features including age, glucose level, BMI, and medical history
- Target variable indicating whether the patient had a stroke

After data cleaning:
- Records with missing or unreliable values were handled
- Irrelevant features such as ID were removed
- Final dataset size reduced after cleaning steps

## Methodology

### Data Cleaning
- Converted invalid BMI values to missing and imputed using median
- Removed rows with unknown smoking status
- Converted categorical variables into factors
- Encoded the target variable as binary

### Feature Engineering
Two additional features were created:
- age_group to capture non-linear age effects
- high_glucose based on a clinical diabetes threshold

### Handling Class Imbalance
The dataset is heavily imbalanced, so ROSE (Random Over-Sampling Examples) was used to balance the training data while keeping the test set untouched.

### Model Training

Two models were trained using 5-fold cross-validation:

1. Logistic Regression  
Used as a baseline model for comparison

2. Random Forest  
Used to capture non-linear relationships and feature interactions

Hyperparameters were tuned using cross-validation.

## Model Performance

The models were evaluated on a held-out test set:

- Logistic Regression AUC: 0.802  
- Random Forest AUC: 0.844  

Random Forest outperformed Logistic Regression across all evaluation metrics :contentReference[oaicite:1]{index=1}.

## Key Insights

- Age is the strongest predictor of stroke risk  
- Average glucose level is highly associated with stroke  
- BMI contributes indirectly through other health factors  
- Handling class imbalance is critical for meaningful results  
- Random Forest captures complex interactions better than linear models  

## Tools and Technologies

- R
- caret
- randomForest
- tidyverse
- ROSE

## Project Structure

- stroke_prediction.R  
- stroke_analysis.Rmd  
- 0-stroke_prediction_report.pdf  

Full report is included in the repository with detailed analysis and results.

## Conclusion

This project demonstrates how machine learning can be applied to healthcare risk prediction. A well-designed model, combined with proper handling of class imbalance and feature engineering, can produce meaningful and actionable insights.

## Contact

Islam Gamal  
LinkedIn: https://www.linkedin.com/in/islamgamalig

Open to opportunities in data science and machine learning.

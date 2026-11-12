Project Structure and Execution Guide
======================================

This directory contains the code and outputs required to reproduce the preprocessing, model selection,
training, and prediction pipeline described in the project report.

For the leaderboard evaluation, two submissions were selected: one generated using the Logistic
Regression model and one generated using the Linear SVC model.

Contents
--------
- main.py
  Main execution script. It runs the complete preprocessing and training pipeline
  and generates the final prediction files for submission.

- Biancone_Nicolotti_report_winter2026.pdf
  Final project report describing the methodology, preprocessing steps, model selection, hyperparameter
  tuning, and experimental results.

- analysis.ipynb
  File containing all our exploratory work

- submission_Logistic_Regression.csv
  Submission file generated using the Logistic Regression model.

- submission_Linear_SVC.csv
  Submission file generated using the Linear Support Vector Classifier

- development.csv

- evaluation.csv


Main Script Overview
--------------------

The main.py script builds the complete machine learning pipeline and trains two classifiers:
	1.	Logistic Regression
	2.	Linear Support Vector Classifier (Linear SVC)

These models correspond to the best-performing configurations identified during the model selection and hyperparameter tuning phase.

For each model, the script performs the following steps:
	-	applies the full preprocessing pipeline;
	-	trains the classifier on the development set;
	-	generates predictions on the evaluation set;
	-	produces a dedicated submission file.

Running Instructions
-------
- Run main.py to generate the submissions file:
	•	submission_Logistic_Regression.csv
	•	submission_Linear_SVC.csv



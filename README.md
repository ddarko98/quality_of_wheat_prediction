Machine Learning for Durum Wheat Quality Prediction
Overview
This project focuses on developing a machine learning model to accurately classify the quality of durum wheat grains based on morphological and textural features. The goal is to distinguish between high-quality ('Vitreous'), softer ('Starchy'), and defective or non-durum ('Foreign') grains, automating a crucial step in agricultural quality assessment.

The project follows a standard machine learning workflow:

Raw Data Gathering 

Exploratory Data Analysis (EDA) 

Data Processing 

Feature Engineering 

Algorithm Selection 

Model Training and Evaluation 

Model Exporting 

Dataset
The model was trained on the "Durum Wheat Dataset" available on Kaggle.


Instances: 9,000 


Features: 237 


Target Classes: The dataset is perfectly balanced with 3,000 samples for each of the three quality categories:



Vitreous: Represents high-quality, hard durum wheat.


Starchy: Represents softer wheat.


Foreign: Includes non-durum or defective grains.

The features are categorized into:


Morphological Features: Characteristics related to the shape and size of the grain, such as AREA, MAJORAXIS, MINORAXIS, and PERIMETER.



Textural Features: Extracted from grain images using Gabor filters, these features capture subtle surface textures that are crucial for distinguishing quality.


The initial exploratory data analysis confirmed that the dataset contains no missing values or duplicate rows.



Methodology
1. Data Processing and Feature Engineering
Column names were standardized to a consistent format (lowercase, no special characters). The categorical 

Target variable was encoded into numerical format using LabelEncoder for model training.

2. Feature Selection
A correlation analysis was performed to identify the most relevant features. All features with an absolute correlation value greater than 0.3 with the encoded target variable were selected for the model. This resulted in a refined set of 

189 important features.

3. Model Training and Evaluation
The data was split into a training set (80%) and a testing set (20%). A 

Pipeline was constructed to first scale the numerical features using StandardScaler and then feed them into a classifier.


Five different classification algorithms were evaluated:

Logistic Regression 

Gaussian Naive Bayes 

Decision Tree 

Random Forest 

Support Vector Machine (LinearSVC) 

Results
The performance of each model was measured by its accuracy on the test set.

Model	Accuracy
Random Forest		
0.98 

Decision Tree	
0.98 

Support Vector Machine	
0.97 

Logistic Regression	
0.96 

Naive Bayes	
0.87 


Export to Sheets
Best Model: Random Forest
The 

Random Forest Classifier was selected as the best model, achieving an accuracy of 98%.

Classification Report:

              precision    recall  f1-score   support

           0       0.99      1.00      1.00       600
           1       0.99      0.96      0.98       600
           2       0.97      0.99      0.98       600

    accuracy                           0.98      1800
   macro avg       0.99      0.98      0.98      1800
weighted avg       0.99      0.98      0.98      1800


Technologies Used
Python

Pandas 

NumPy 

Matplotlib 

Seaborn 

Scikit-learn 

Joblib 

How to Use
The final trained model pipeline was saved to a file named wheat_sorting.pkl. You can load this model and use it to predict the quality of new wheat samples.

Python

import joblib
import pandas as pd
import numpy as np

# Load the trained pipeline
pipeline = joblib.load("../model/wheat_sorting.pkl")

# Load the LabelEncoder to decode predictions
# Note: The original LabelEncoder should be saved alongside the model for this to work.
# For this example, we'll assume classes are encoded as: Foreign=0, Starchy=1, Vitreous=2
le = LabelEncoder()
le.classes_ = np.array(['Foreign', 'Starchy', 'Vitreous']) # Example, use the actual fitted encoder

# Create a sample DataFrame with the 189 important features
# The columns must be in the same order as used in training.
# Replace with your actual data.
X_train_columns = [...] # Load the list of 189 column names from the project
n_samples = 5
test_data = pd.DataFrame(
    np.random.rand(n_samples, 189) * 100,
    columns=X_train_columns
)

# Make predictions
predictions_encoded = pipeline.predict(test_data)

# Decode the predictions to get the original labels
predicted_labels = le.inverse_transform(predictions_encoded)

print("Test Samples:\n", test_data.head())
print("\nPredicted Wheat Quality:\n", predicted_labels)

# Import necessary libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



# Read the heart disease data from CSV file

df = pd.read_csv('Data_Science_and_Machine_learning_JosePortilla\\DATA\\heart.csv')



# Take a peek at the first few rows of the data

print(df.head())



# Get basic information about the data (data types, missing values, etc.)

print(df.info())



# Get summary statistics of the data (mean, standard deviation, etc.)

print(df.describe().transpose())



# Count the number of patients with and without heart disease

print(df['target'].value_counts())



# Visualize the distribution of heart disease (presence vs. absence)

sns.countplot(data=df, x='target')

plt.show()  # Explicitly display the plot



# Explore relationships between features and heart disease using pairplots

# Coloring by target highlights differences between groups

sns.pairplot(data=df[['age', 'trestbps', 'chol', 'thalach', 'target']], hue='target')

plt.show()  # Explicitly display the plot



# Calculate correlation coefficients between features

print(df.corr())



# Create a heatmap to visualize correlations (stronger correlations are darker)

plt.figure(figsize=(10, 6), dpi=100)

sns.heatmap(df.corr(), annot=True, cmap='viridis')

plt.show()  # Explicitly display the plot



# Separate features (X) and target variable (y)

X = df.drop('target', axis=1)

y = df['target']



# Display the first few rows of features and target variable

print(X.head(3))

print(y.head(3))



# Split data into training and testing sets for model evaluation (10% for testing)

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=101)



# Standardize features (scale them to have a mean of 0 and standard deviation of 1)

# This can improve the performance of some machine learning models

from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()

scaled_X_train = scaler.fit_transform(X_train)

scaled_X_test = scaler.transform(X_test)



# Create a logistic regression model (suitable for binary classification)

from sklearn.linear_model import LogisticRegression



log_model = LogisticRegression(solver='saga', max_iter=5000, multi_class='ovr')  # One-vs-Rest strategy



# Define a grid of hyperparameters to tune (regularization type, penalty strength, etc.)

penalty = ['l1', 'l2', 'elasticnet']

l1_ratio = np.linspace(0, 1, 20)

C = np.logspace(0, 10, 20)

param_grid = {'penalty': penalty, 'l1_ratio': l1_ratio, 'C': C}



# Use GridSearchCV to find the best hyperparameter combination based on training data

from sklearn.model_selection import GridSearchCV



grid_model = GridSearchCV(estimator=log_model, param_grid=param_grid)

grid_model.fit(scaled_X_train, y_train)



# Predict class probabilities for the test set (useful for calculating ROC AUC)

y_pred_proba = grid_model.predict_proba(scaled_X_test)



# Predict class labels (0 or 1) for the test set

y_pred = grid_model.predict(scaled_X_test)



# Evaluate model performance using accuracy score

from sklearn.metrics import accuracy_score



print("Accuracy:", accuracy_score(y_test, y_pred))



# Create a confusion matrix to visualize model predictions vs. actual labels

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay



cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:\n", cm)



# Display the confusion matrix in a heatmap format

ConfusionMatrixDisplay(cm).plot()

plt.show()  # Explicitly display the plot



# Classification Report: Precision, Recall, F1-score, Support

from sklearn.metrics import classification_report



print("Classification Report:\n", classification_report(y_test, y_pred))



# ROC Curve: Visualize model performance for different classification thresholds

from sklearn.metrics import roc_curve, PrecisionRecallDisplay



fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:, 1]) # Consider probability for class 1 (heart disease)

plt.plot(fpr, tpr, label='ROC Curve')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC Curve for Heart Disease Classification')

plt.legend()

plt.grid(True)

plt.show()



# Precision-Recall Curve: Trade-off between precision and recall

PrecisionRecallDisplay.from_predictions(y_test, y_pred)

plt.show()



# Predicting for a new patient (assuming the patient data is stored in a variable named 'patient')

patient = [[54.0, 1.0, 0.0, 122.0, 286.0, 0.0, 0.0, 116.0, 1.0, 3.2, 1.0, 2.0, 2.0]]

prediction = grid_model.predict(patient)



if prediction[0] == 0:

 print("Predicted Class: No Heart Disease")

else:

 print("Predicted Class: Heart Disease")



# Predicting probability of heart disease for the new patient

probability = grid_model.predict_proba(patient)[0][1]

print("Probability of Heart Disease:", probability)

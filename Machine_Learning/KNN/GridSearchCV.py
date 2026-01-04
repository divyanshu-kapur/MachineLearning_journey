#Import the required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the sonar data set
df = pd.read_csv('.../DATA/sonar.all-data.csv')

# Explore the data (get a basic understanding)

df.head()  # View the first few rows

df.info()  # Check data types and missing values

df.describe()  # Summarize numerical data

# Visualize feature correlations
print("**Heatmap of Feature Correlations (excluding label):")
plt.figure(figsize=(8, 4), dpi=150)
sns.heatmap(df.drop('Label', axis=1).corr(), cmap='coolwarm')
plt.show()  # Display the heatmap

# Class distribution (number of rocks and mines)
df['Label'].value_counts().sort_values(ascending=False)  # Count occurrences of each class

# Create a new target variable with numerical labels (0 for Rock, 1 for Mine)
df['target'] = df['label'].map({'R': 0, 'M': 1})
df.head()  # View the first few rows again

# Identify top 5 correlated features with the target variable
top_correlations = np.abs(df.drop('Label', axis=1).corr()['target']).nlargest(6).sort_values()
print(top_correlations)

# Data Splitting (training and testing sets)
# Use 90% of data for training and 10% for testing with a fixed random state for reproducibility
X = df.drop(['target', 'Label'], axis=1)  # Features for the model
y = df['Label']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Machine Learning Pipeline with KNN
# This pipeline combines feature scaling and KNN classification

# Preprocessing step: Standardization
scaler = StandardScaler()

# KNN model
knn = KNeighborsClassifier()

# Combine steps into a pipeline
operations = [('scaler', scaler), ('knn', knn)]
pipe = Pipeline(operations)

# Hyperparameter Tuning with GridSearchCV
# Test different K values (number of neighbors) and find the best performing model

# Define a list of K values to explore
k_values = list(range(1, 30))

# Create a parameter grid for the KNN model (n_neighbors)
paramgrid = {'knn__n_neighbors': k_values}

# GridSearchCV object with pipeline, parameter grid, 5-fold cross-validation, and accuracy scoring
grid_model = GridSearchCV(estimator=pipe, param_grid=paramgrid, cv=5, scoring='accuracy')

# Train the model on the training data
grid_model.fit(X_train, y_train)

# Print the parameters of the best performing model found by GridSearchCV
print("**Best Model Parameters:")
grid_model.best_estimator_.get_params()

# Get the mean test scores for each K value during cross-validation
scores = grid_model.cv_results_['mean_test_score']

# Plot the relationship between K values and accuracy
plt.plot(k_values, scores)
plt.xlabel('K values (Number of Neighbors)')
plt.ylabel('Accuracy')
plt.show()  # Display the plot

# Model Evaluation on Test Set
# Use the best model from GridSearchCV to make predictions on the unseen test data

# Make predictions on the test set
pred = grid_model.predict(X_test)

# Import necessary functions for evaluation metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Calculate accuracy score
accuracy = accuracy_score(y_test, y_pred)
print("**Accuracy:", accuracy)

# Generate confusion matrix to understand model performance on each class
confusion_matrix(y_test, y_pred)
print("**Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Classification report provides detailed metrics like precision, recall, and F1 score

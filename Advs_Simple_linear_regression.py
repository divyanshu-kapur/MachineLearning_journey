# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the advertising data from CSV file
df = pd.read_csv('C:\\Users\\DELL\\Advertising.csv')

# Take a peek at the first few rows of the data (optional)
df.head()

# Create subplots for visualizing relationships between features and sales
fig, axes = plt.subplots(figsize=(12, 4), nrows=1, ncols=3, dpi=200)

# Scatter plot - TV vs Sales
axes[0].plot(df['TV'], df['sales'], 'o', color='green')
axes[0].set_xlabel('TV')  # Label x-axis
axes[0].set_ylabel('Sales')  # Label y-axis

# Scatter plot - Radio vs Sales
axes[1].plot(df['radio'], df['sales'], 'o', color='blue')
axes[1].set_xlabel('Radio')
axes[1].set_ylabel('Sales')

# Scatter plot - Newspaper vs Sales
axes[2].plot(df['newspaper'], df['sales'], 'o', color='red')
axes[2].set_xlabel('Newspaper')
axes[2].set_ylabel('Sales')

plt.tight_layout()

# Separate features (X) and target variable (y)
X = df.drop('sales', axis=1)  # All columns except 'sales' are features
y = df['sales']  # 'sales' is the target variable

# Split data into training and testing sets for model evaluation
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# Create a linear regression model
from sklearn.linear_model import LinearRegression
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the testing data
test_predictions = model.predict(X_test)

# Calculate evaluation metrics - Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE)
from sklearn.metrics import mean_absolute_error, mean_squared_error
mae = mean_absolute_error(y_test, test_predictions)
rmse = np.sqrt(mean_squared_error(y_test, test_predictions))

print("Mean Absolute Error (MAE):", mae)
print("Root Mean Squared Error (RMSE):", rmse)

# Calculate test residuals (difference between actual and predicted sales)
test_residuals = y_test - test_predictions

# Visualize test residuals
sns.scatterplot(x=y_test, y=test_residuals)
plt.axhline(color='r', ls='--', lw=1.8)  # Add a horizontal line at zero for reference

# Train a final model on all data (for visualization)
final_model = LinearRegression()
final_model.fit(X, y)

# Get regression coefficients (impact of each feature on sales)
final_model_coef = final_model.coef_
print("Regression Coefficients:", final_model_coef)

# Predict sales using the final model
y_pred = final_model.predict(X)

# Visualize predicted sales vs actual sales for all features (TV, Radio, Newspaper)
fig, axes = plt.subplots(figsize=(12, 4), nrows=1, ncols=3, dpi=200)

axes[0].plot(df['TV'], df['sales'], 'o')  # Actual sales
axes[0].plot(df['TV'], y_pred, 'o')  # Predicted sales
axes[0].set_xlabel('TV')
axes[0].set_ylabel('Sales')

axes[1].plot(df['radio'], df['sales'], 'o')
axes[1].plot(df['radio'], y_pred, 'o')
axes[1].set_xlabel('Radio')
axes[1].set_ylabel('Sales')

axes[2].plot(df['newspaper'], df['sales'], 'o')
axes[2].plot(df['newspaper'], y_pred, 'o')
axes[2].set_xlabel('Newspaper')
axes[2].set_ylabel('Sales')

plt.tight_layout()

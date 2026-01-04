import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the Ames Housing data
df = pd.read_csv("...\\Ames_Housing_Data.csv")

df.head()

df.info()

df.describe()

#Lets check the correlation between the columns
df1 = df.select_dtypes(exclude='object')
df1.corr()['SalePrice'].sort_values()

# Scatter plots - key features vs SalePrice
sns.scatterplot(data=df,x='Overall Qual',y='SalePrice')

sns.scatterplot(data=df,x='Garage Area',y='SalePrice')

sns.scatterplot(data=df,x='Overall Cond',y='SalePrice')

sns.scatterplot(data=df,x='Gr Liv Area',y='SalePrice')

#Remove the outliers
drop_index = df[(df['Gr Liv Area']>4000) & (df['SalePrice']<300000)]
df = df.drop(drop_index.index,axis=0)
sns.scatterplot(data=df,x='Gr Liv Area',y='SalePrice')

#Lets load description file of the ames_housing_dataset
with open("...Ames_Housing_Feature_Description.txt",'r') as f:
    print(f.read())

#Feature engineering

#Drop the 'PID' column (may not be useful for prediction)
df = df.drop('PID',axis=1)
df.isnull().sum()

#Check the %age of the missing data

def percent_missing(df):
	percent_nan = 100*df.isnull().sum() / len(df)
	percent_nan = percent_nan[percent_nan>0].sort_values()

	return percent_nan

percent_nan = percent_missing(df)
percent_nan

# Visualize missing data percentages (limit y-axis for better view of low percentages)
plt.figure(figsize=(8,5))
sns.barplot(x = percent_nan.index,y=percent_nan)
plt.xticks(rotation=90)
plt.show()

plt.figure(figsize=(8,5))
sns.barplot(x = percent_nan.index,y=percent_nan)
plt.xticks(rotation=90)
# check for the features with null values percent less than 1
plt.ylim(0,1)
plt.show()

percent_nan[percent_nan < 1]

df[df['Electrical'].isnull()]['Bsmt Unf SF']
df[df['Bsmt Half Bath'].isnull()]['Bsmt Full Bath']

df = df.dropna(axis=0,subset = ['Electrical','Garage Cars'])

percent_nan = percent_missing(df)
percent_nan

plt.figure(figsize=(8,5))
sns.barplot(x=percent_nan.index,y= percent_nan)
plt.xticks(rotation=90)
# check for the features with null values percent less than 1
plt.ylim(0,1)
plt.show()

bsmt_str_cols = ['Bsmt Qual','Bsmt Cond','Bsmt Exposure','BsmtFin Type 1','BsmtFin Type 2']
bsmt_num_cols = ['BsmtFin SF 1','BsmtFin SF 2','Bsmt Unf SF','Total Bsmt SF','Bsmt Full Bath','Bsmt Half Bath']

df.loc[:,bsmt_num_cols] = df[bsmt_num_cols].fillna(0)
df.loc[:,bsmt_str_cols] = df[bsmt_str_cols].fillna('None')

percent_nan = percent_missing(df)
percent_nan

df['Mas Vnr Type'] = df['Mas Vnr Type'].fillna('None')
df['Mas Vnr Area'] = df['Mas Vnr Area'].fillna(0)

percent_nan = percent_missing(df)
percent_nan

plt.figure(figsize=(8,5))
sns.barplot(x=percent_nan.index,y= percent_nan)
plt.xticks(rotation=90)
plt.show()

garage_cols = ['Garage Type','Garage Finish','Garage Qual','Garage Cond']
df[garage_cols] = df[garage_cols].fillna('None')

plt.figure(figsize=(8,5))
sns.barplot(x=percent_nan.index,y= percent_nan)
plt.xticks(rotation=90)

plt.show()

df['Garage Yr Blt'] = df['Garage Yr Blt'].fillna(0)

df = df.drop(['Pool QC','Misc Feature','Alley','Fence'],axis=1)

percent_nan = percent_missing(df)

plt.figure(figsize=(8,5))
sns.barplot(x=percent_nan.index,y= percent_nan)
plt.xticks(rotation=90)
plt.show()

df['Fireplace Qu'] = df['Fireplace Qu'].fillna('None')

df['Lot Frontage'].info()

df[df['Fireplace Qu'].isnull()]

#Lot Frontage is linked to Neighborhood. So lets explore relation between the both
plt.figure(figsize=(8,10),dpi=200)
sns.boxplot(x='Lot Frontage',y='Neighborhood',data=df)

#add mean values in the missing values
df.groupby('Neighborhood')['Lot Frontage'].mean()
df['Lot Frontage'] = df.groupby('Neighborhood')['Lot Frontage'].apply(lambda value:value.fillna(value.mean()))

df[df['Lot Frontage'].isnull()]['Neighborhood']

df['Lot Frontage'] = df['Lot Frontage'].fillna(0)

#One Hot Encoding

df.head()

# Convert 'MS SubClass' to string to avoid encoding issues during one-hot encoding
df['MS SubClass'] = df['MS SubClass'].apply(str)

# Separate data into numerical and categorical features
df_object = df.select_dtypes(include='object')
df_numeric = df.select_dtypes(exclude='object')

# One-hot encode categorical features (drop first category to avoid dummy variable trap)
df_get_dummies = pd.get_dummies(df_object,drop_first=True,dtype=int)
df_get_dummies.head()

# Combine encoded categorical features with numerical features
final_df = pd.concat([df_numeric,df_get_dummies],axis=1)
final_df.head()

# Analyze correlation between features and Sale Price
final_df.corr()['SalePrice'].sort_values()
final_df.shape

#Split the data into training and test set
X = final_df.drop('SalePrice')
y = final_df['SalePrice']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=101)

# Standardize features using StandardScaler (improves model convergence)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
scaled_X_train = scaler.transform(X_train)
scaled_X_test = scaler.transform(X_test)

# Define an ElasticNet model (regularized linear regression with L1 and L2 penalties)
from sklearn.linear_model import ElasticNet
elastic_model = ElasticNet(max_inter = 100000)

# Define hyperparameter grid for grid search (alpha and l1_ratio)
param_grid = {'alpha':[0.1,0.5,1,5,10,100],'l1_ratio':[0.1,0.3,0.5,0.8,1]}

# Perform grid search cross-validation to find optimal hyperparameters for ElasticNet
from sklearn.model_selection import GridSearchCV

grid_model = GridSearchCV(estimator = elastic_model,scoring='neg_mean_squared_error',cv=5,verbose=1,param_grid=param_grid)
grid_model.fit(scaled_X_train,y_train)

# Make predictions on the testing set using the best model from grid search
y_pred = grid_model.predict(scaled_X_test)

# Evaluate model performance using mean absolute error (MAE) and mean squared error (MSE)
from sklearn.metrics import mean_absolute_error, mean_squared_error
mean_absolute_error(y_test,y_pred)
mean_squared_error(y_test,y_pred)

# Calculate average Sale Price for reference
df['SalePrice'].mean()

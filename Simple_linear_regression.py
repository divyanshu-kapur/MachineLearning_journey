import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('C:\\Users\\DELL\\Advertising.csv')
df.head()

fig,axes= plt.subplots(figsize=(12,4),nrows=1,ncols=3,dpi=200)

axes[0].plot(df['TV'],df['sales'],'o',color='green')
axes[0].set_xlabel('TV')
axes[0].set_ylabel('Sales')

axes[1].plot(df['radio'],df['sales'],'o',color='blue')
axes[1].set_xlabel('Radio')
axes[1].set_ylabel('Sales')

axes[2].plot(df['newspaper'],df['sales'],'o',color='red')
axes[2].set_xlabel('newspaper')
axes[2].set_ylabel('Sales')

plt.tight_layout();

#Assign features and label(X and y)
X = df.drop('sales',axis=1)
y = df['sales']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

from sklearn.linear_model import LinearRegression 

model = LinearRegression()
model.fit

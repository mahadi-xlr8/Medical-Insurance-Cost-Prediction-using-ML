import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

dataSet=pd.read_csv("insurance.csv")
print(dataSet.head())
print("rows :",dataSet.shape[0],"columns :",dataSet.shape[1])
print("\nData Set Info\n")
dataSet.info()

print("\n\n",dataSet.describe())

# distribution of age value
sns.set()
plt.figure(figsize=(6,6))
sns.histplot(dataSet['age'])
plt.title('Age Distribution')
plt.show()

# Gender column
plt.figure(figsize=(6,6))
sns.countplot(x='sex', data=dataSet)
plt.title('Sex Distribution')
plt.show()

# bmi distribution
plt.figure(figsize=(6,6))
sns.histplot(dataSet['bmi'])
plt.title('BMI Distribution')
plt.show()
#Normal BMI Range --> 18.5 to 24.9

# children column
plt.figure(figsize=(6,6))
sns.countplot(x='children', data=dataSet)
plt.title('Children')
plt.show()

print("\nChildrens",dataSet['children'].value_counts(),sep="\n")

# smoker column
plt.figure(figsize=(6,6))
sns.countplot(x='smoker', data=dataSet)
plt.title('smoker')
plt.show()

print("\nSmokers",dataSet["smoker"].value_counts(),sep="\n")

# region column
plt.figure(figsize=(6,6))
sns.countplot(x='region', data=dataSet)
plt.title('region')
plt.show()

print("\nRegions",dataSet["region"].value_counts(),sep="\n")

# distribution of charges value
plt.figure(figsize=(6,6))
sns.histplot(dataSet['charges'])
plt.title('Charges Distribution')
plt.show()

# encoding sex column
dataSet.replace({'sex':{'male':0,'female':1}}, inplace=True)

# encoding 'smoker' column
dataSet.replace({'smoker':{'yes':0,'no':1}}, inplace=True)

# encoding 'region' column
dataSet.replace({'region':{'southeast':1,'southwest':2,'northeast':3,'northwest':4}}, inplace=True)

print(dataSet.head)

X = dataSet.drop(columns='charges', axis=1)
Y = dataSet['charges']

#Splitting the data into Training data & Testing Data

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# loading the Linear Regression model
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# prediction on training data
training_data_prediction =regressor.predict(X_train)

# R squared value
r2_train = metrics.r2_score(Y_train, training_data_prediction)
print('R squared vale : ', r2_train)

# prediction on test data
test_data_prediction =regressor.predict(X_test)

# R squared value
r2_test = metrics.r2_score(Y_test, test_data_prediction)
print('R squared vale : ', r2_test)

### end of training and testing process ###








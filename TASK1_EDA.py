#import Necessary files
import kagglehub # for dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Store data into variable
data = pd.read_csv("Titanic-Dataset.csv")

#Read first 5 Rows
print(data.head(5))
#Read last 5 Rows
print(data.tail(5))
#Print data types
print(data.dtypes)

#Remove un-necessary rows
data = data.drop(['SibSp','Parch','Embarked', 'Pclass'],axis=1)

#Check if any duplicate data exists
duplicated_data = data[data.duplicated()]
print(duplicated_data.shape)

#Read first 5 Rows
print(data.head(5))
print(data.count())

#Remove Duplicate Data
data= data.drop_duplicates();

#Check if Null data Exists
print(data.isnull().sum())
data.dropna()

#Check Outliers

# Assuming 'data' is your DataFrame
numeric_data = data.select_dtypes(include='number')  # Work with numeric columns only

# Calculate Q1, Q3, and IQR
Q1 = numeric_data.quantile(0.25)
Q3 = numeric_data.quantile(0.75)
IQR = Q3 - Q1
print(IQR)

# Calculate bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Remove outliers
cleaned_data = numeric_data[~((numeric_data < lower_bound) | (numeric_data > upper_bound)).any(axis=1)]

# If you want the full DataFrame back (including non-numeric columns)
data = data.loc[cleaned_data.index]


print(data)


#Visualization
#Gender Diversity
data['Sex'].value_counts().plot(kind='bar', color=['blue', 'pink'])
plt.xlabel("Sex")
plt.ylabel("Sex Count")
plt.show()

#Age Diversity
data['Age'].value_counts().plot(kind='hist', bins=40)
plt.xlabel("Age")
plt.ylabel("Age Count")
plt.show()

#Fare Range
data['Fare'].value_counts().plot(kind='hist', bins=30)
plt.xlabel("Fare")
plt.ylabel("Fare frequency")
plt.show()

#HeatMap
sns.heatmap(numeric_data.corr(), cmap='hot', annot=True)
plt.show()
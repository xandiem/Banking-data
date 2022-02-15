# %% 1. Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sqlalchemy import true
#Load data into pandas for usage


data = pd.read_csv('bank.csv')
print(data.head(5))
print(data.info())

# %% 2. Exploration of the data
#2.1. Look at percentage of data marked as fraud

fraud = data['fraud'].value_counts()
print(type(fraud))
ax1 = fraud.plot.bar(x='fraud', y='amount', ylabel='Number of cases', xlabel='Fraud')

# %%
#2.2. Look at percentage of 'fraud cases by age'
fraud_by_age = data.groupby(['fraud','age']).size()
print(fraud_by_age)

grouped = data.groupby(data.fraud)

non_fraudulent = grouped.get_group(0)
fraudulent = grouped.get_group(1)

print(non_fraudulent)
print(fraudulent)

count_n_fraud = non_fraudulent.groupby(['age']).count()
count_fraud = fraudulent.groupby(['age']).count()

print(count_n_fraud)
print(count_fraud)

percentages = 100*(count_fraud / (count_n_fraud+count_fraud))
sel_cols = percentages['step']
percent = sel_cols.copy()
print(percent)

ax = percent.plot.bar(x='age', y='step', rot=0, ylabel='Cases considered fraud (%)', xlabel='Age category')

# %%
#Fraud by gender

count_n_fraud = non_fraudulent.groupby(['gender']).count()
count_fraud = fraudulent.groupby(['gender']).count()

percentages = 100* (count_fraud / (count_n_fraud+count_fraud))
sel_cols = percentages['step']
percent = sel_cols.copy()
print(percent)

ax = percent.plot.bar(x='gender', y='step', ylabel='Fraud cases out of total (%)')

# %% Fraud by category
print(data['category'].unique())
count_n_fraud = non_fraudulent.groupby(['category']).count()
count_fraud = fraudulent.groupby(['category']).count()

print(count_n_fraud)
print(count_fraud)

percentages = 100* (count_fraud / (count_n_fraud+count_fraud))
sel_cols = percentages['step']
percent = sel_cols.copy()
print(percent)

ax = percent.plot.bar(x='category', y='step', ylabel='Fraud cases out of total (%)')

#Typical amount transferred, particularly for fraud cases
# %%
avg_amount = data['amount'].mean()
print(avg_amount)

avg_n_fraud = non_fraudulent['amount'].mean()
avg_fraud = fraudulent['amount'].mean()

print(avg_n_fraud)
print(avg_fraud)

ax = plt.bar(x=[0, 1], height=[avg_n_fraud, avg_fraud], color=['b', 'g'])
plt.title='A graph showing the difference is transaction size for fraudulent and non transactions'
# %%
#Now pre-processing for ML usage
#print(data.head(5))
#print(data.info())

#No null values but several objects to deal with

#Remove customer id
new_data = data.drop(columns='customer')
#print(data.head(10))
#print(data.info)

#print(data['age'])
le = LabelEncoder()
new_data['age'] = le.fit_transform(new_data['age'])
new_data['gender'] = le.fit_transform(new_data['gender'])

#new_data['zipcodeOri'] = le.fit_transform(new_data['zipcodeOri'])
#new_data['merchant'] = le.fit_transform(new_data['merchant'])
#new_data['zipMerchant'] = le.fit_transform(new_data['zipMerchant'])
new_data['category'] = le.fit_transform(new_data['category'])

print(data['zipMerchant'].unique())
print(new_data.head(10))
# %% Machine learning algos

X = new_data.iloc[:, 0:7]
Y = new_data.iloc[:, 8]
print(X)
print(Y)

train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2)
print(test_X)
scores = cross_val_score(XGBRegressor(), train_X, train_Y, cv=5)
print(scores)
# %%

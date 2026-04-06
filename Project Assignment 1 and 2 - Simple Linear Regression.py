#!/usr/bin/env python
# coding: utf-8

# # PROBLEM 1: Delivery Time Prediction

# In[13]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


# In[2]:


df = pd.read_csv("C:/Users/udiit/Downloads/delivery_time.csv")
df.head()


# In[5]:


df.info()
df.describe()

#NOTE: No missing values found. Dataset is clean and ready for analysis.


# In[6]:


df.isnull().sum()
#  good (no cleaning needed)


# # Exploratory Data Analysis (EDA)

# In[7]:


plt.scatter(df['Sorting Time'], df['Delivery Time'])
plt.xlabel("Sorting Time")
plt.ylabel("Delivery Time")
plt.title("Sorting Time vs Delivery Time")
plt.show()

#INSIGHT - #1. Helps visualize relationship


# In[8]:


#Correlation - 

df.corr()


# In[9]:


#Histogram - 

df.hist()
plt.show()


# # Model Building (Linear Regression)

# In[10]:


# Defining Variables - 
X = df[['Sorting Time']]   # Independent variable
y = df['Delivery Time']    # Dependent variable


# In[11]:


#Train Model - 

model = LinearRegression()
model.fit(X, y)


# In[12]:


#Predictions

y_pred = model.predict(X)


# In[14]:


#Model Evaluation - R² Score

r2 = r2_score(y, y_pred)
print("R2 Score:", r2)


# In[15]:


##Model Evaluation - RMSE 
rmse = np.sqrt(mean_squared_error(y, y_pred))
print("RMSE:", rmse)


# In[16]:


#Visualization (Regression Line)

plt.scatter(X, y)
plt.plot(X, y_pred, color='red')
plt.xlabel("Sorting Time")
plt.ylabel("Delivery Time")
plt.title("Regression Line")
plt.show()


# In[17]:


# Transformation -
X_log = np.log(df[['Sorting Time']])
y_log = np.log(df['Delivery Time'])

model_log = LinearRegression()
model_log.fit(X_log, y_log)

y_pred_log = model_log.predict(X_log)

print("R2 Score (Log Model):", r2_score(y_log, y_pred_log))


# In[18]:


# # NEW PREDICTION # # - -

new_data = pd.DataFrame({'Sorting Time': [5, 10, 15]})

predictions = model.predict(new_data)

new_data['Predicted Delivery Time'] = predictions
new_data


# In[19]:


# SAVING Prediction - 
new_data.to_csv("delivery_predictions.csv", index=False)

#A Simple Linear Regression model was built to predict delivery time based on sorting time.
#The model showed a strong positive relationship between variables.
#After evaluating using R² and RMSE, the best model was selected.
#Predictions were generated successfully for new input values.


# # Problem 2: Salary Prediction

# In[20]:


df_salary = pd.read_csv("C:/Users/udiit/Downloads/Salary_Data.csv")
df_salary.head()


# In[21]:


# Data Understanding - 

df_salary.info()
df_salary.describe()


# In[22]:


#Data cleaning - checking null/missing values - 

df_salary.isnull().sum()


# In[23]:


# EDA - Scatterplot - 

plt.scatter(df_salary['YearsExperience'], df_salary['Salary'])
plt.xlabel("Years of Experience")
plt.ylabel("Salary Hike")
plt.title("Experience vs Salary")
plt.show()


# In[24]:


# 2  - Correlation - 
df_salary.corr()


# In[25]:


#3 . Distribution - 

df_salary.hist()
plt.show()


# # Model Building

# In[27]:


# Defining Variables
X = df_salary[['YearsExperience']]
y = df_salary['Salary']


# In[28]:


# Train Models - 
model_salary = LinearRegression()
model_salary.fit(X, y)


# In[31]:


# Predictions 
y_pred_salary = model_salary.predict(X)


# In[32]:


# Model Evaluation - 
r2_salary = r2_score(y, y_pred_salary)
print("R2 Score:", r2_salary)


# In[33]:


#RMSE - 

rmse_salary = np.sqrt(mean_squared_error(y, y_pred_salary))
print("RMSE:", rmse_salary)


# In[34]:


# Visualization - 
plt.scatter(X, y)
plt.plot(X, y_pred_salary, color='red')
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Regression Line")
plt.show()


# In[35]:


# TRANSFORMATION - #1. Log Transformation 

X_log = np.log(df_salary[['YearsExperience']])
y_log = np.log(df_salary['Salary'])

model_log = LinearRegression()
model_log.fit(X_log, y_log)

y_pred_log = model_log.predict(X_log)

print("R2 Score (Log Model):", r2_score(y_log, y_pred_log))


# # NEW PREDICTIONS

# In[36]:


new_exp = pd.DataFrame({'YearsExperience': [2, 5, 10]})

predictions_salary = model_salary.predict(new_exp)

new_exp['Predicted Salary'] = predictions_salary
new_exp


# In[37]:


#Saving file 

new_exp.to_csv("salary_predictions.csv", index=False)

#NOTE: "A Linear Regression model was built to predict salary based on years of experience.
#The model showed a strong positive linear relationship.
#Evaluation metrics (R² and RMSE) indicate high model accuracy.
#Predictions were successfully generated for new experience values."


# In[ ]:





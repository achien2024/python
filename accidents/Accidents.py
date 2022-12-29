#!/usr/bin/env python
# coding: utf-8

# # Introduction

# Hello, my name is Aaron Chien and I am using the 'Industrial Safety and Health Analytics Database' from Kaggle as an introductury project on python. The CSV file includes accidents that have occured in difference countries and includes multiple columns. In this project, I would analyze these accidents and see what we can do to predict these accidents.

# Let's view and clean the data

# In[1]:


import sklearn
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

import statsmodels.api as sm
import statsmodels.formula.api as smf

import seaborn as sns 

import matplotlib.pyplot as plt 

import pandas as pd

import numpy as np

import math

import pickle

import dbm

import os

from sklearn.impute import SimpleImputer
import sys

from statistics import mode

import datetime

from functools import partial, reduce


# In[2]:


# timeline of accident
# figure out which plants and which countries and great map of it showing average accident occurances 
# ratio of men and women, types of employees, etc
# figure out predicted accident level from potential accident level, lowest slope is the safest 


# In[3]:


sys.path.append("/Users/aaron/Accident")


# In[4]:


indust = pd.read_csv("industrial.csv")


# In[5]:


indust.head()


# Since some of these column names are in spanish, let's translate those into spanish!

# In[6]:


indust.rename(columns = {"Employee ou Terceiro" : 'Employee_or_Third_Party', 'Risco Critico' : 'Critical_Risk',
                         'Industry Sector' : 'Industry', 'Accident Level' : 'Accident_Level',
                         'Potential Accident Level' : 'Potential_Accident_Level', 'Data' : 'Date', 
                         'Genre' : 'Gender'},
              inplace = True)


# In[7]:


indust.head()


# Now let's see what unique values are in each separate column to see which columns are interesting.

# In[8]:


print(indust.Countries.unique());
print(indust.Local.unique());
print(indust.Industry.unique()); 
print(indust.Employee_or_Third_Party.unique()); 
print(indust.Critical_Risk.unique());
print(indust.Gender.unique());
print(indust.Accident_Level.unique())


# Now that we know what unique values there are, let's convert the Date column into a date object

# In[9]:


indust['Date'] = pd.to_datetime(indust['Date'])


# We need to turn the roman numerals into numbers for easier manipulation!

# In[10]:


# turn roman numerals into numbers up to 5
def roman_num(x):
    y = []
    for i in range(0,len(x)) :
        if x[i] == "I":
            y.append(1)
        elif x[i] == "II":
            y.append(2)
        elif x[i] == "III":
            y.append(3)
        elif x[i] == "IV":
            y.append(4)
        else:
            y.append(5)
    return(y)


# In[11]:


'''Would this work?
def roman_num(x):
    for i in range(0,len(x)) :
        if x[i] == "I":
            x[i] = 1
        elif x[i] == "II":
            x[i] = 2
        elif x[i] == "III":
            x[i] = 3
        elif x[i] == "IV":
            x[i] = 4
        else:
            x[i] = 5
    return(x)
'''


# Converting Accident levels

# In[12]:


n1 = roman_num(indust["Accident_Level"]) 
n2 = roman_num(indust["Potential_Accident_Level"])


# In[13]:


indust.drop(['Potential_Accident_Level', 'Accident_Level'], axis = 1, inplace = True)


# In[14]:


indust['Accident_Level'] = n1
indust['Potential_Accident_Level'] = n2
indust['Avoided_Accident_Level'] = indust['Potential_Accident_Level'] - indust['Accident_Level']


# Creating months column in the data 

# In[15]:


month = pd.DatetimeIndex(indust["Date"]).month.astype(str) 
year = pd.DatetimeIndex(indust["Date"]).year.astype(str)
slash = np.repeat("/", 439)
indust['months'] = list(reduce(partial(map, str.__add__), (month, slash, year)))


# In[16]:


indust.head()


# # Creating Timeline of Accidents per Month

# Creating data frame of the mean of the accident levels for each month for clean graphing

# In[17]:


timeline = indust.groupby("months")[['Accident_Level', 'Potential_Accident_Level','Avoided_Accident_Level']].mean().reset_index()


# In[18]:


timeline['months'] = pd.to_datetime(timeline['months'])
timeline = timeline.sort_values("months")
timeline.head()


# In[19]:


plt.figure(figsize=(20, 10))
plt.plot(timeline['months'], timeline['Accident_Level'], label = 'Accident Level')
plt.plot(timeline['months'], timeline['Potential_Accident_Level'], label = 'Potential Accident Level')
plt.legend()
plt.show()


# This is a timeline of accidents base on the average per month. However, we can see that there is no exact relationship between time and accidents, so we should examine the columns more closely. 

# In[20]:


plt.figure(figsize=(20, 10))
plt.plot(timeline['months'], timeline['Avoided_Accident_Level'], label = 'Avoided Accident Level', 
         color = "green")
plt.legend()


# The avoided accident level, the difference between the accident and the potential accident, is plotted here against time.

# # Exploring Gender

# In[21]:


fig = plt.figure(figsize=(20, 10))
gs = fig.add_gridspec(1, 3)

ax = fig.add_subplot(gs[0, 0])
sns.violinplot(x = "Gender", y = "Accident_Level", data = indust, split = True)
ax.set_xlabel("Accident Level")

ax = fig.add_subplot(gs[0, 1])
sns.violinplot(x = "Gender", y = "Potential_Accident_Level", data = indust, split = True)
ax.set_xlabel("Potential Accident Level")

ax = fig.add_subplot(gs[0, 2])
sns.violinplot(x = "Gender", y = "Avoided_Accident_Level", data = indust, split = True)
ax.set_xlabel("Avoided Accident Level")

fig.tight_layout()
plt.show()


# Examining the violin plots here, we can see that the accident levels for both females and males are more clustered around level 1, but the males are more distributed and widespread than the female. However, when we examine the the potential accident levels for the males and females, the female is more clustered around a level of 2, meaning that the females were more exposed to low level potential accidents compared to the men, which had a median of 3 and a clustering around 4. But it is important to say that the male potential accident was also heavily distributed. A good idea to point out is that we only had 22 female samples, compared to 419 male samples, so perhaps looking at gender is not a good idea as we need more sample to really come to a dense conclusion if males on average are more exposed to higher level potential accidents compared to the female. Since the data is skewed, it's best to examine the median as the average. 

# In[22]:


gender_acc = indust.groupby("Gender")[['Accident_Level', 'Potential_Accident_Level', 'Avoided_Accident_Level']].median().reset_index()
gender_acc


# In[23]:


gender = indust.groupby("Gender")[['Accident_Level']].count().reset_index()

gender_perct = [round(i, 3) for i in [gender['Accident_Level'][0] / sum(gender['Accident_Level']),
                                      gender['Accident_Level'][1] / sum(gender['Accident_Level'])]]

print(gender)
pd.DataFrame({'Percentage': gender_perct}, index = ['Female', 'Male'])


# We can see that the femal gender only occupies 5% of the dataset so it's harder to explore this part of the data as we only have 22 females in the total 429 sample. In order to see if gender plays a rolw in accidents, we would need more data.

# In[24]:


gen = gender_acc.T
gen.drop("Gender", axis = 0, inplace = True)
gen.rename(columns = {0 : 'Female', 1 : 'Male'}, inplace = True)
gen.plot.bar()
plt.show()


# # Exploring Industry

# In[25]:


fig = plt.figure(figsize=(20, 10))
gs = fig.add_gridspec(1, 3)

ax = fig.add_subplot(gs[0, 0])
sns.violinplot(x = "Industry", y = "Accident_Level", data = indust, split = True)
ax.set_xlabel("Accident Level")

ax = fig.add_subplot(gs[0, 1])
sns.violinplot(x = "Industry", y = "Potential_Accident_Level", data = indust, split = True)
ax.set_xlabel("Potential Accident Level")

ax = fig.add_subplot(gs[0, 2])
sns.violinplot(x = "Industry", y = "Avoided_Accident_Level", data = indust, split = True)
ax.set_xlabel("Avoided Accident Level")

fig.tight_layout()
plt.show()


# Observing the violin plots, we can see that all the industry have a median of 1 and clusters at 1 for the accident level. We can also see that it is right-skewed, so like before, using the median as the average would be better. Examining the Potential Accident Level, we can see that Mining has a median around a level 4, Metals has a median around 3, and Others has a median around 1. It is important to point out that in the metals industry, they seem to be evenly cluster around 2, 3, and 4. We can see that the potential accident level are more well-spread but the actual accident levels for all the industrys are not as heavily spread and more tied to level 1. 

# In[26]:


sns.pairplot(indust, hue = 'Industry')


# Observing the kernel density plots, we can see that the Mining industry has the highest density of higher potential accident levels, but metals have the widest distribution. We can also see that the accident level is right skewed for each industry, so it's better to use the median as the count of where most of the data lies. 

# In[27]:


indust.groupby("Industry")[['Accident_Level', 'Potential_Accident_Level', 'Avoided_Accident_Level']].median().reset_index()


# In[28]:


indust.groupby("Industry")[['Accident_Level']].count().reset_index()


# In[29]:


indust_reg = smf.ols(formula = 'Accident_Level ~ Potential_Accident_Level*Industry', data = indust)
# metal_reg = sm.OLS(metals['Accident_Level'], metals['Potential_Accident_Level']).fit()
indust_reg.fit().summary()


# Examining the metal industry, the equation of the linear equation is:
# 
# $Accident Level = 0.0020 + -0.1375\times Mining + 0.5247 \times Others + (0.4729 + 0.0485 \times Mining + -0.0116 \times Others) \times Potential Accident Level$

# In[30]:


plt.figure(figsize = (20, 10))
x = np.array(range(6))
y_metals = 0.0020 + 0.4729 * x
y_mining = -0.1355 + 0.5214 * x
y_others = 0.5267 + 0.4613 * x

plt.plot(x, y_metals, label = "Metal Industry")
plt.plot(x, y_mining, label = "Mining Industry")
plt.plot(x, y_others, label = "Others")

plt.title("Industry Accident Prediction from Potential")
plt.xlabel("Potential Accident Level")
plt.ylabel("Accident Level")

plt.legend()
plt.show()


# Observing the linear regression, we can see that others occur higher accident levels given a potential accident level and mining is the least. However, mining has the steepest slope, indicating that they will gain higher accident levels as the potential accident level increases. From this graph, we can conclude that although others may incur higher accidents, mining has a trade off between safety and damages. 

# # Exploring Employee or Third Party

# In[31]:


fig = plt.figure(figsize=(20, 10))
gs = fig.add_gridspec(1, 3)

ax = fig.add_subplot(gs[0, 0])
sns.violinplot(x = "Employee_or_Third_Party", y = "Accident_Level", data = indust, split = True)
ax.set_xlabel("Accident Level")

ax = fig.add_subplot(gs[0, 1])
sns.violinplot(x = "Employee_or_Third_Party", y = "Potential_Accident_Level", data = indust, split = True)
ax.set_xlabel("Potential Accident Level")

ax = fig.add_subplot(gs[0, 2])
sns.violinplot(x = "Employee_or_Third_Party", y = "Avoided_Accident_Level", data = indust, split = True)
ax.set_xlabel("Avoided Accident Level")

fig.tight_layout()
plt.show()


# Observing the violin plots for the different types of employees, we can see that it is still right skewed for the accident level at level 1. The potnetial accident level are more widespread however but seems to be a bit more clustered around 4 but with a median of 3. The avoided accident level is interesting for Third Party as all three employee status had relatvely the same median, but for the third party, it has a median of 1 compared to 2 for the others. 

# In[32]:


sns.pairplot(indust, hue = 'Employee_or_Third_Party')


# Observing the kernel density graph of the employee status, we can see that the third party had a wide spread and distribution of potential accident levels, which explains the peciular different median the third party had compared to the different employee statuses. 

# In[33]:


indust.groupby("Employee_or_Third_Party")[['Accident_Level', 'Potential_Accident_Level', 'Avoided_Accident_Level']].median().reset_index()


# In[34]:


indust.groupby("Employee_or_Third_Party")[['Accident_Level']].count().reset_index()


# In[35]:


employee_reg = smf.ols(formula = 'Accident_Level ~ Potential_Accident_Level*Employee_or_Third_Party', data = indust)
# metal_reg = sm.OLS(metals['Accident_Level'], metals['Potential_Accident_Level']).fit()
employee_reg.fit().summary()


# Examining the employee industry, the equation of the linear equation is:
# 
# $Accident Level = 0.2209 + -0.0812\times Third Party + 0.0669 \times Remote Third Party + (0.3947 + 0.1020 \times Third Party + -0.0258 \times Remote Third Party) \times Potential Accident Level$

# In[36]:


plt.figure(figsize = (20, 10))
x = np.array(range(6))
y_employee = 0.2209 + 0.3947 * x
y_thirdparty = 0.1397 + 0.4967 * x
y_remotethirdparty = 0.2878 + 0.3689 * x

plt.plot(x, y_employee, label = "Employee")
plt.plot(x, y_thirdparty, label = "Third Party")
plt.plot(x, y_remotethirdparty, label = "Remote Third Party")

plt.title("Employee Status Accident Prediction from Potential")
plt.xlabel("Potential Accident Level")
plt.ylabel("Accident Level")

plt.legend()
plt.show()


# ## Exploring Countries

# In[37]:


fig = plt.figure(figsize=(20, 10))
gs = fig.add_gridspec(1, 3)

ax = fig.add_subplot(gs[0, 0])
sns.violinplot(x = "Countries", y = "Accident_Level", data = indust, split = True)
ax.set_xlabel("Accident Level")

ax = fig.add_subplot(gs[0, 1])
sns.violinplot(x = "Countries", y = "Potential_Accident_Level", data = indust, split = True)
ax.set_xlabel("Potential Accident Level")

ax = fig.add_subplot(gs[0, 2])
sns.violinplot(x = "Countries", y = "Avoided_Accident_Level", data = indust, split = True)
ax.set_xlabel("Avoided Accident Level")

fig.tight_layout()
plt.show()


# Observing the violin plots, we can see that all three countries cluster around an accident level of 1, but the potential accident level tells a different story. We see that Country 1 has a median potential accident level of 4, Country 2 has it at 3, and Country 3 has it at 1. We also see that Country 1 and Country 2 are exposed to a wider distribution of potential accidents. 

# In[38]:


sns.pairplot(indust, hue = 'Countries')


# Examining the kernel density of the three countries, we can see that country 1 are more exposed to higher potential accident levels but country 2 has a wider distribution. We can also wee that the accident level for each country is indeed right-skewed, so it's best to use the median as an average.

# In[39]:


indust.groupby("Countries")[['Accident_Level', 'Potential_Accident_Level', 'Avoided_Accident_Level']].median().reset_index()


# In[40]:


indust.groupby("Countries")[['Accident_Level']].count().reset_index()


# In[41]:


country_reg = smf.ols(formula = 'Accident_Level ~ Potential_Accident_Level*Countries', data = indust)
# metal_reg = sm.OLS(metals['Accident_Level'], metals['Potential_Accident_Level']).fit()
country_reg.fit().summary()


# Examining the countries, the equation of the linear equation is:
# 
# $Accident Level = -0.2548 + 0.4977\times Country 2 + 0.7604 \times Country 3 + (0.5597 + -0.1695 \times Country 2 + -0.0562 \times Country 3) \times Potential Accident Level$

# In[42]:


plt.figure(figsize = (20, 10))
x = np.array(range(6))
y_country1 = -0.2548 + 0.5597 * x
y_country2 = 0.2429 + 0.3902 * x
y_country3 = 0.5056 + 0.5035 * x

plt.plot(x, y_country1, label = "Country 1")
plt.plot(x, y_country2, label = "Country 2")
plt.plot(x, y_country3, label = "Country 3")

plt.title("Country Accident Prediction from Potential")
plt.xlabel("Potential Accident Level")
plt.ylabel("Accident Level")

plt.legend()
plt.show()


# Exploring countries, we can see that Country 3 have a higher accident level given a potential accident level and country 1 have the steepest slope. Meaning that although Country 1 may have a smaller accident level given a potential level, there is a higher rate of actual accident level as the potential accident level increases.

# ## Conclusion

# After examiing all the linear regression, we can conclude that the best relationship to predict accidents from potential accidents is the countries one. Out of all the linear regression we have performed, it has the highest R-squared value out of all of it at 0.296. We can also examine the t values from the linear regression summary that at the 10% level, most of the coefficients are significant enough to not be 0. The most accurate equation would be:
# 
# $Accident Level = -0.2548 + 0.4977\times Country 2 + 0.7604 \times Country 3 + (0.5597 + -0.1695 \times Country 2 + -0.0562 \times Country 3) \times Potential Accident Level$

# I would reccomend that we need more columns in order to create a better linear regression model to predict accidents from potential accidents and other categorical variables that may help. I can only hypothesize that the different Countries linear regression fits the best because different countries have different policies on safety, and some may be more lenient than others. In general, I feel this dataset is a bit incomplete and more research and data gathering would need to be down. Data such as helmet wear, fatigue level, age, medical issues, or hours worked would all work and may also shed more light on what is causing these accidents. 

#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, chi2_contingency, f_oneway


# In[3]:


data = pd.read_csv('Titanic-Dataset.csv')


# In[4]:


male_survival = data[data['Sex'] == 'male']['Survived']
female_survival = data[data['Sex'] == 'female']['Survived']


# In[5]:


t_stat, p_value = ttest_ind(male_survival, female_survival, nan_policy='omit')
print(f"T-test: t-statistic = {t_stat}, p-value = {p_value}")


# In[6]:


contingency_table = pd.crosstab(data['Pclass'], data['Survived'])
chi2, p, dof, expected = chi2_contingency(contingency_table)
print(f"Chi-Square Test: chi2 = {chi2}, p-value = {p}")


# In[7]:


data['AgeGroup'] = pd.cut(data['Age'], bins=[0, 18, 35, 60, 100], labels=['Child', 'Youth', 'Adult', 'Senior'])
anova_result = f_oneway(
    data[data['AgeGroup'] == 'Child']['Survived'],
    data[data['AgeGroup'] == 'Youth']['Survived'],
    data[data['AgeGroup'] == 'Adult']['Survived'],
    data[data['AgeGroup'] == 'Senior']['Survived']
)
print(f"ANOVA: F-statistic = {anova_result.statistic}, p-value = {anova_result.pvalue}")


# In[ ]:





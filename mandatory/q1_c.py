#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv("/Users/kodalialekhya/Downloads/15_new.csv")


# In[4]:


df = df.drop(columns='Unnamed: 0')


# In[5]:


df.head()


# In[6]:


data = df[(df.Date>'2020-09-30') & (df.Date<'2021-01-01')]
data.head()


# In[11]:


def get_xy(x):

  n = len(x)
  x = sorted(x)
  x_cdf = []
  y_cdf = []
  y_curr = 0

  x_cdf.append(0)
  y_cdf.append(0)

  for i in x:
    y_curr += 1/n
    y_cdf.append(y_curr)
    x_cdf.append(i)

  return x_cdf,y_cdf


# In[13]:


def draw_ecdf(x1, y1, x2, y2, max_diff, max_ind):
    plt.figure(figsize=(20,10))
    plt.step(x1, y1, where="post", label="CDF-D1")
    plt.step(x2, y2, where="post", label="CDF-D2")
    plt.yticks(np.arange(0, 1.1, 1/10))
    plt.title("Empirical CDF")
    plt.xlabel("Sample Points")
    plt.ylabel("Pr[X<x]")
    plt.scatter([max_ind],[0], color='red', marker='x', s=100, label=f'Max Diff {max_diff} at {max_ind}')
    plt.grid(which="both")
    plt.legend()
    plt.show()


# In[35]:


def ks_1_sample_test(data1,data2, statement, threshold=0.05):
  x1, y1 = get_xy(data1)

  n = len(data2)

  diff=[]
  for i in range(n):
    diff.append( np.absolute( y1[i] - data2[i]  ) )

  max_diff = np.max(diff)

  max_ind = np.argmax(diff)

  if max_diff > threshold:
    print(f"Max value = {max_diff} > C: {threshold}, We reject H0: "+statement)
  else:
    print(f"Max value = {max_diff} <= C: {threshold}, We reject H0: "+statement)


# In[36]:


mean_nd_confirmed = np.mean(data['nd_confirmed_per_day'].values)
mean_nd_deaths = np.mean(data['nd_deaths_per_day'].values)


# In[37]:


def calc_poisson(param, x):
  return stats.poisson.cdf(x, param)


# In[38]:


x1, y1 = get_xy(data['ne_confirmed_per_day'].values)
val = calc_poisson(mean_nd_confirmed, x1)
ks_1_sample_test(data['ne_confirmed_per_day'].values, val, "The cases follow poisson distrubtion")


# Result of 1 sample KS test for last 3 months of 2020 for NE state cases with poisson distribution
# 
# Null Hypothesis (H0):
# 
# Distribution of last 3 months of 2020 for NE state cases equals poisson distribution
# 
# Alternate Hypothesis (H1):
# 
# Distribution of last 3 months of 2020 for NE state cases not equals poisson distribution
# 
# Procedure:
# 
# We have obtained parameters for poisson distribution by using MME on last 3 months for ND state data. We have taken the c = 0.05 and n=92 as given in documentation and calculated the maximum difference of the CDF of the distributions at all the points.
# 
# Result: 
# 
# As the KS test value is 0.913 which is greater than 0.05 we are rejecting the NULL hypothesis.

# In[18]:


x1, y1 = get_xy(data['ne_deaths_per_day'].values)
val = calc_poisson(mean_nd_deaths, x1)
ks_1_sample_test(data['ne_deaths_per_day'].values, val, "The deaths follow poisson distrubtion")


# Result of 1 sample KS test for last 3 months of 2020 for NE state deaths with poisson distribution
# 
# Null Hypothesis (H0):
# 
# Distribution of last 3 months of 2020 for NE state deaths equals poisson distribution
# 
# Alternate Hypothesis (H1):
# 
# Distribution of last 3 months of 2020 for NE state deaths not equals poisson distribution
# 
# Procedure:
# 
# We have obtained parameters for poisson distribution by using MME on last 3 months for ND state data. We have taken the c = 0.05 and n=92 as given in documentation and calculated the maximum difference of the CDF of the distributions at all the points.
# 
# Result: 
# 
# As the KS test value is 0.497 which is greater than 0.05 we are rejecting the NULL hypothesis.

# In[40]:


def calc_geometric(param, x):
  return stats.geom.cdf(x, param)


# In[44]:


x1, y1 = get_xy(data['ne_confirmed_per_day'].values)
val = calc_poisson(1/mean_nd_confirmed, x1)
ks_1_sample_test(data['ne_confirmed_per_day'].values, val, "The cases follow geometric distrubtion")


# Result of 1 sample KS test for last 3 months of 2020 for NE state cases with geometric distribution
# 
# Null Hypothesis (H0):
# 
# Distribution of last 3 months of 2020 for NE state cases equals geometric distribution
# 
# Alternate Hypothesis (H1):
# 
# Distribution of last 3 months of 2020 for NE state cases not equals geometric distribution
# 
# Procedure:
# 
# We have obtained parameters for geometric distribution by using MME on last 3 months for ND state data. We have taken the c = 0.05 and n=92 as given in documentation and calculated the maximum difference of the CDF of the distributions at all the points.
# 
# Result: 
# 
# As the KS test value is 0.996 which is greater than 0.05 we are rejecting the NULL hypothesis.

# In[45]:


x1, y1 = get_xy(data['ne_deaths_per_day'].values)
val = calc_poisson(1/mean_nd_deaths, x1)
ks_1_sample_test(data['ne_deaths_per_day'].values, val, "The cases follow geometric distrubtion")


# Result of 1 sample KS test for last 3 months of 2020 for NE state deaths with geometric distribution
# 
# Null Hypothesis (H0):
# 
# Distribution of last 3 months of 2020 for NE state deaths equals geometric distribution
# 
# Alternate Hypothesis (H1):
# 
# Distribution of last 3 months of 2020 for NE state deaths not equals geometric distribution
# 
# Procedure:
# 
# We have obtained parameters for geometric distribution by using MME on last 3 months for ND state data. We have taken the c = 0.05 and n=92 as given in documentation and calculated the maximum difference of the CDF of the distributions at all the points.
# 
# Result: 
# 
# As the KS test value is 0.772 which is greater than 0.05 we are rejecting the NULL hypothesis.

# In[46]:


def calc_binomial(n, p, x):
  return stats.binom.cdf(x, n, p)


# In[23]:


var_nd_confirmed = np.var(data['nd_confirmed_per_day'].values)
n = (mean_nd_confirmed * mean_nd_confirmed)/(mean_nd_confirmed - var_nd_confirmed)
p = mean_nd_confirmed/n

x1, y1 = get_xy(data['ne_confirmed_per_day'].values)
val = calc_binomial(n, p, x1)
ks_1_sample_test(data['ne_confirmed_per_day'].values, val, "The cases follow binomial distrubtion")


# Result of 1 sample KS test for last 3 months of 2020 for NE state cases with binomial distribution
# 
# Null Hypothesis (H0):
# 
# Distribution of last 3 months of 2020 for NE state cases equals binomial distribution
# 
# Alternate Hypothesis (H1):
# 
# Distribution of last 3 months of 2020 for NE state cases not equals binomial distribution
# 
# Procedure:
# 
# We have obtained parameters for binomial distribution by using MME on last 3 months for ND state data. We have taken the c = 0.05 and n=92 as given in documentation and calculated the maximum difference of the CDF of the distributions at all the points.
# 
# Result: 
# 
# As the KS test value is 1 which is greater than 0.05 we are rejecting the NULL hypothesis.

# In[24]:


var_nd_deaths = np.var(data['nd_deaths_per_day'].values)
n = (mean_nd_deaths * mean_nd_deaths)/(mean_nd_deaths - var_nd_deaths)
p = mean_nd_deaths/n

x1, y1 = get_xy(data['ne_deaths_per_day'].values)
val = calc_binomial(n, p, x1)
ks_1_sample_test(data['ne_deaths_per_day'].values, val, "The cases follow binomial distrubtion")


# Result of 1 sample KS test for last 3 months of 2020 for NE state deaths with binomial distribution
# 
# Null Hypothesis (H0):
# 
# Distribution of last 3 months of 2020 for NE state deaths equals binomial distribution
# 
# Alternate Hypothesis (H1):
# 
# Distribution of last 3 months of 2020 for NE state deaths not equals binomial distribution
# 
# Procedure:
# 
# We have obtained parameters for binomial distribution by using MME on last 3 months for ND state data. 
# We have taken the c = 0.05 and n=92 as given in documentation and calculated the maximum difference of the CDF of the distributions at all the points.
# 
# Result: 
# 
# As the KS test value is 1 which is greater than 0.05 we are rejecting the NULL hypothesis.

# In[62]:


def ks_2_sample_test(data1,data2, threshold=0.05, draw=True):
  x1, y1 = get_xy(data1)
  x2, y2 = get_xy(data2)

  n = int(min([max(x1),max(x2)])) +10

  y1_all = []
  temp=0
  for i in np.arange(n):
    ind = np.where(np.array(x1) == i)[0]
    if len(ind)==0:
      y1_all.append(temp)
    else:
      y1_all.append(y1[ind[-1]])
      temp = y1[ind[-1]]

  y2_all = []
  temp=0
  for i in np.arange(n):
    ind = np.where(np.array(x2) == i)[0]
    if len(ind)==0:
      y2_all.append(temp)
    else:
      y2_all.append(y2[ind[-1]])
      temp = y2[ind[-1]]

  diff=[]
  for i in range(n):
    diff.append( np.absolute( y1_all[i] - y2_all[i]  ) )

  max_diff = np.max(diff)

  max_ind = np.argmax(diff)

  if draw:
    draw_ecdf(x1,y1,x2,y2, max_diff, max_ind)

  if max_diff > threshold:
    print(f"Max value = {max_diff} > C: {threshold}, We reject H0")
  else:
    print(f"Max value = {max_diff} <= C: {threshold}, We reject H0")


# In[63]:


ks_2_sample_test(data['nd_confirmed_per_day'].values, data['ne_confirmed_per_day'].values)


# Result of 2 sample KS test for last 3 months of 2020 for ND state and NE state cases
# 
# Null hypothesis (H0):
#     
# Distribution of ND state cases equals distribution of NE state cases
# 
# Alternate hypothesis(H1):
#     
# Distribution of ND state cases not equals to distribution of NE state cases
# 
# Procedure :
#     
# We have taken c = 0.05 as given in documentation and calculated the maximum difference of the CDF of the distributions at all the points.
# 
# Result:
#     
# As the KS test value is 0.75 which is greater than 0.05 we are rejecting the NULL hypothesis.

# In[27]:


ks_2_sample_test(data['nd_deaths_per_day'].values, data['ne_deaths_per_day'].values)


# Result of 2 sample KS test for last 3 months of 2020 for ND state and NE state deaths
# 
# Null hypothesis (H0):
# 
# Distribution of ND state deaths equals distribution of NE state deaths
# 
# Alternate hypothesis(H1):
# 
# Distribution of ND state deaths not equals to distribution of NE state deaths
# 
# Procedure :
# 
# We have taken the c = 0.05 as given in documentation and calculated the maximum difference of the CDF of the distributions at all the points.
# 
# Result:
# 
# As the KS test value is 0.467 which is greater than 0.05 we are rejecting the NULL hypothesis.

# In[60]:


def permutation_test(X, Y, n=1000, threshold=0.05):
  T_obs = abs(np.mean(X) - np.mean(Y))
  xy = np.append(X,Y)
  p_value = 0.0
  T = []
  for i in range(n):
    permutation = np.random.permutation(xy)
    X1 = permutation[:len(X)]
    Y1 = permutation[len(X):]
    Ti = abs(np.mean(X1) - np.mean(Y1))
    T.append(Ti)

  T = np.array(T)
  p_value = np.sum(T>T_obs)/len(T)
  print("The p-value is: ", p_value)
  if (p_value <= threshold):
    print("p-value less than or equal to the threshold ==> Reject the Null Hypothesis")
  else:
    print("p-value greater than threshold ==> Accept the Null Hypothesis")


# In[61]:


x1 = data['nd_confirmed_per_day'].values
x2 = data['ne_confirmed_per_day'].values

y1=data['nd_deaths_per_day'].values
y2=data['ne_deaths_per_day'].values

permutation_test(x1,x2)
permutation_test(y1,y2)


# Result of Permutation test for last 3 months of 2020 for ND state and NE state cases
# 
# Null hypothesis (H0):
# 
# Distribution of ND state cases equals distribution of NE state cases
# 
# Alternate hypothesis(H1):
# 
# Distribution of ND state cases not equals distribution of NE state cases
# 
# Result:
# 
# As the Permutation test value is 0.0 which is less than 0.05 we are rejecting the NULL hypothesis.

# Result of Permutation test for last 3 months of 2020 for ND state and NE state deaths
# 
# Null hypothesis (H0):
# 
# Distribution of ND state deaths equals distribution of NE state deaths
# 
# Alternate hypothesis(H1):
# 
# Distribution of ND state deaths not equals distribution of NE state deaths
#  
# Result:
# 
# As the Permutation test value is 0.003 which is less than 0.05 we are rejecting the NULL hypothesis.

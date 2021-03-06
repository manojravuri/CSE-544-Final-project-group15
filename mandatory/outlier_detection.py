# -*- coding: utf-8 -*-
"""Outlier_detection.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1i3vkiBjQZxzfrDoElcE5Z1BeYlFxybGI
"""

from google.colab import drive
drive.mount('/content/gdrive')

# Commented out IPython magic to ensure Python compatibility.
# %cd gdrive/MyDrive/probstat

import pandas as pd
import numpy as np

def get_daily_counts(x):
  y=np.array([0])
  y=np.append(y,x)
  y=y[:-1]
  return x-y

def outliers_and_data_cleaning(x):
  Q1 = np.percentile(x, 25, interpolation = 'midpoint')
  Q3 = np.percentile(x, 75, interpolation = 'midpoint')
  #tukey's rule outliers are with first_quartile-1.5IQR and third_quartile+1.5IQR
  IQR=Q3-Q1
  left=Q1-1.5*IQR
  right=Q3+1.5*IQR
  mean=np.mean(x)
  print("Lower bound",left)
  print("Upper bound",right)
  #replace with mean if the outliers are found
  for i in range(x.size):
    if(x[i]<left and x[i]!=0):
      x[i]=mean
    elif(x[i]>right and x[i]!=0):
      x[i]=mean
  return x

"""In the above function, we apply tukey's rule to identify the left and right bounds for the data


"""

def remove_outliers_and_create_new_file():
    dataframe = pd.read_csv('../15.csv')
    ne_confirmed = dataframe['NE confirmed'].to_numpy()
    nd_confirmed = dataframe['ND confirmed'].to_numpy()
    ne_deaths = dataframe['NE deaths'].to_numpy()
    nd_deaths = dataframe['ND deaths'].to_numpy()

    #get the daily counts from each state and each category
    ne_confirmed_per_day = get_daily_counts(ne_confirmed)
    nd_confirmed_per_day = get_daily_counts(nd_confirmed)
    ne_deaths_per_day = get_daily_counts(ne_deaths)
    nd_deaths_per_day = get_daily_counts(nd_deaths)

    #remove outliers by replacing them with mean and not changing zeros
    print("nd_confirmed_per_day")
    nd_confirmed_per_day = outliers_and_data_cleaning(nd_confirmed_per_day)
    print("ne_confirmed_per_day")
    ne_confirmed_per_day = outliers_and_data_cleaning(ne_confirmed_per_day)
    print("nd_deaths_per_day")
    nd_deaths_per_day = outliers_and_data_cleaning(nd_deaths_per_day)
    print("ne_deaths_per_day")
    ne_deaths_per_day = outliers_and_data_cleaning(ne_deaths_per_day)

    #append to a new file
    dataframe['ne_confirmed_per_day'] = ne_confirmed_per_day
    dataframe['nd_confirmed_per_day'] = nd_confirmed_per_day
    dataframe['ne_deaths_per_day'] = ne_deaths_per_day
    dataframe['nd_deaths_per_day'] = nd_deaths_per_day
    dataframe.to_csv('../15_updated.csv')

"""In the above the function we take each column data, find the outliers, replace with the mean of rhe data and output to a new file"""

remove_outliers_and_create_new_file()

"""**Approach**


*   We used tukey's rule to find the lower and upper bounds where outliers exists 
*   Since removing all the outliers may cause gaps in the time series data, we replace the outliers with the mean data
* changing the outliers does not cause any changes to the data distribution.





"""


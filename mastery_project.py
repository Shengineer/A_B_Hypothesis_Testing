# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 13:39:22 2023

@author: kit
"""

import pandas as pd
from scipy.stats import t
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Load the CSV file into a pandas DataFrame
file_path = r'C:\Users\kit\OneDrive\Documents\Masterschool Docs\mastery_project.csv'
raw_df = pd.read_csv(file_path)

# Replace null values with zeros
raw_df.fillna(0.0, inplace=True)
# print(raw_df.head())

# Setup control and treatment variants of the dataframe
control_df = raw_df[raw_df['cust_group'] == 'A']
treatment_df = raw_df[raw_df['cust_group'] == 'B']

# Calculate the average spent by each user for both variants
def avg_spent(x1):
    total_spent = x1.groupby('users_id')['spent'].sum()         # Grouping by user allows all purchases by one user to be summed 
    average_spent = total_spent.mean()
    return average_spent

# Calculate the average spent by each user for both variants
# Control
average_spent_control = avg_spent(control_df)
print("Control Average: " + str(average_spent_control))
# Treatment
average_spent_treatment = avg_spent(treatment_df)
print("Treatment Average: " + str(average_spent_treatment) + "\n")

# Count the number of users from each country in the control and treatment dataframes
control_countries = control_df.groupby('country')['users_id'].count().sort_values(ascending=True)
treatment_countries = treatment_df.groupby('country')['users_id'].count().sort_values(ascending=True)

# Create a horizontal bar chart for the control and treatment dataframes
plt.figure(1, figsize=(10,5))
plt.subplot(121)
control_countries.plot(kind='barh', color='lightblue')
plt.title('Control')
plt.ylabel('Country')
plt.xlabel('Number of Users')
for i, v in enumerate(control_countries):
    plt.text(v + 5, i - 0.1, str(v), color='black', fontweight='bold', ha='right', va='center')

plt.subplot(122)
treatment_countries.plot(kind='barh', color='lightgreen')
plt.title('Treatment')
plt.ylabel('Country')
plt.xlabel('Number of Users')
for i, v in enumerate(treatment_countries):
    plt.text(v + 5, i - 0.1, str(v), color='black', fontweight='bold', ha='right', va='center')

plt.tight_layout()
plt.show()

#Hypothesis test, Null hypothesis the averages are equal
# Assuming unequal variance, then unpooled standard error for test
def welch_ttest(x1, x2, col_name, alternative):
    
    n1 = len(x1)
    n2 = len(x2)
    
    m1 = avg_spent(x1)
    m2 = avg_spent(x2)
    
    v1 = np.var(x1[col_name], ddof=1)
    v2 = np.var(x2[col_name], ddof=1)
    
    unpooled_se = np.sqrt(v1/n1 + v2/n2)
    delta = m2 - m1                         # Treatment mean - Control mean
    
    df = (v1/n1 + v2/n2)**2 / ((v1/n1)**2/(n1-1) + (v2/n2)**2/(n2-1))
    tstat = delta /  unpooled_se
    
    # Default is two side t-test
    if alternative == "less":
        p = t.cdf(tstat, df)
    elif alternative == "greater":
        p = 1 - t.cdf(tstat, df)
    else:
        p = 2 * t.cdf(-abs(tstat), df)
    
    # upper and lower bounds
    lb = delta - t.ppf(0.975,df)*unpooled_se 
    ub = delta + t.ppf(0.975,df)*unpooled_se
  
    return pd.DataFrame(np.array([tstat,df,p,delta,lb,ub]).reshape(1,-1),
                         columns=['T statistic','df','pvalue 2 sided','Difference in mean','lb','ub'])

# Get the T-statistics outputs from this function
print(welch_ttest(control_df, treatment_df, 'spent', "unequal"))
print("\n")

# Calculate the 95% confidence interval for the average amount spent per user
def one_sample_t_ci(sample_df, col_name, alpha=0.05):
    """
    Calculates the confidence interval for the population mean of a specified column in a DataFrame,
    using a one-sample t-interval.
    
    Parameters:
        sample_df (DataFrame): The DataFrame containing the data
        col_name (str): The name of the column for which to calculate the confidence interval
        alpha (float): The significance level for the confidence interval (default=0.05)
        
    Returns:
        A tuple containing the lower and upper bounds of the confidence interval
    """
    total_spent = sample_df.groupby('users_id')['spent'].sum()      # Average spent per user
    sample_mean = total_spent.mean()
    sample_std = sample_df[col_name].std(ddof=1)  # Use ddof=1 for unbiased estimate of sample standard deviation
    n = len(sample_df)-1
    std_error = sample_std / (n ** 0.5)
    sample_df_t = n - 1
    ci = stats.t.interval(1 - alpha, sample_df_t, loc=sample_mean, scale=std_error)
    return ci

# 95% confidence interval for Control
ci_control = one_sample_t_ci(control_df, 'spent')
print("95% confidence interval for Control: " + str(round(ci_control[0], 3)) + ", " + str(round(ci_control[1], 3)))
# 95% confidence interval for Treatment
ci_treatment = one_sample_t_ci(treatment_df, 'spent')
print("95% confidence interval for Treatment: " + str(round(ci_treatment[0], 3)) + ", " + str(round(ci_treatment[1], 3)) + "\n")

# User conversion rate
def conversion_rate(x1):
    '''
    Calculates the conversion rate of a dataframe dependant on the 'spent' column and grouped by user
    
    Parameters:
        x1 (DataFrame): The DataFrame containing the data of df

    Returns:
        x1_converted : Returns the length of the sorted dataframe
        x1_rate : Percentage conversion rate rounded to 2 d.p. of df
    '''
    x1_converted = len(x1.where(x1['spent'] > 0).groupby('users_id')['spent'].sum())        # Combining a where and groupby clause for 'spent' > 0 meaning converted
    x1_rate = round((x1_converted / len(x1)) * 100, 2)                                      # Divide by total and *100 for percentage
    return x1_converted, x1_rate

con_A = conversion_rate(control_df)
print('Control conversion rate = ' + str(con_A[1]) + '%')
con_B = conversion_rate(treatment_df)
print('Treatment conversion rate = ' + str(con_B[1]) + '%' + "\n")

# Confidence interval for conversion rate using normal distribution
def conversion_rate_ci(df):
    '''
    Calculates the 95% Confidence Interval for the conversion rate using the normal distribution
    Parameters:
        df (DataFrame): The DataFrame containing the data

    Returns:
        ci : A tuple containing the lower and upper bounds of the confidence interval
    '''
    p = conversion_rate(df)[0] / len(df)                                                # Calculating sample proportion     
    n = len(df)
    se = np.sqrt(p * (1-p) / n)                                                         # Standard error of the proportion
    z = stats.norm.ppf(0.975)
    ci = round((p - z*se), 5), round((p + z*se), 5)                                     # Rounding the Confidence Interval to 5 d.p.
    return ci

ci_con_control = conversion_rate_ci(control_df)
print("95% confidence interval of conversion rate for CONTROL group using normal distribution:")
print(ci_con_control)
print("\n")
ci_con_treatment = conversion_rate_ci(treatment_df)
print("95% confidence interval of conversion rate for TREATMENT group using normal distribution:")
print(ci_con_treatment)
print("\n")

def two_sample_ztest_prop(x1, x2):
    '''
    Performs a 2 sample, two sided hypothesis test to see if there is a difference in conversion rate between the groups
    Parameters:
        x1 (DataFrame): The DataFrame containing the data of first df
        x2 (DataFrame): The DataFrame containing the data of second df

    Returns:
        z : Z-score of the hypothesis test
        p : The p-value of the hypothesis test
        lb : Lower bound of confidence interval
        up: Upper bound of confidence interval
    '''
    n1 = len(x1)
    n2 = len(x2)
    
    x1_converted = conversion_rate(x1)[0]                       # Hypothesis test is based on conversion rate so needs to be recalculated
    x2_converted = conversion_rate(x2)[0]                       # Used the function created earlier
    
    p1 = x1_converted / n1                                      # Proportion of first df
    p2 = x2_converted / n2                                      # Proportion of second df
    pc = (x1_converted + x2_converted) / (n1 + n2)              # Calculate pooled proportion
    
    z = (p1 - p2) / np.sqrt(pc * (1 - pc) * (1/n1 + 1/n2))      # Test statistic calc, Z-score
    p = 2 * (1 - stats.norm.cdf(abs(z)))                        # And p-value
    
    # Confidence Interval calc
    se = np.sqrt(p1*(1-p1)/n1 + p2*(1-p2)/n2)                   # Unpooled standard error
    
    z = stats.norm.ppf(0.975)                                   # Critical Z value
    
    lb = (p2 - p1) - z * se                                     # Lower and upper bound calculations
    ub = (p2 - p1) + z * se
    
    return z, p, lb, ub

z, p, lb, ub = two_sample_ztest_prop(control_df, treatment_df)
print("z-score:", z)
print("p-value:", p)
print("\n")

print("95% confidence interval for the difference in conversion rate between groups:")
print("Lower bound:", lb)
print("Upper bound:", ub)




import warnings
warnings.simplefilter("ignore")

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss

import warnings
warnings.simplefilter("ignore")

#Url for data used
url ='https://raw.githubusercontent.com/rjafari979/Time-Series-Analysis-and-Moldeing/master/tute1.csv'

df = pd.read_csv(url)
df.rename( columns={'Unnamed: 0':'Date'}, inplace=True )

df['Date'] = pd.date_range(start='Mar-1981', end='Jan-2006', freq='3M')

def plot_data(x, y, data, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(x, y, data=data)
    ax.set(xlabel= xlabel, ylabel=ylabel, title=title)
    ax.grid(True)
    ax.legend()

#Sales over Time
plot_data('Date','Sales',data=df, title='Sales Vs. Time(Years)',xlabel= 'Date', ylabel='Sales (USD)')
plt.savefig('./Images/Sales_Vs_Time(Years).png')

#AdBudget over Time
plot_data('Date','AdBudget', data=df, title='AdBudget Vs. Time(Years)', xlabel='Year', ylabel='AdBudget (USD)')
plt.savefig('./Images/AdBudget_Vs_Time(Years).png')

#GDP over Time
plot_data('Date','GDP', data=df, title='GDP Vs. Time(Years)', xlabel='Year', ylabel='GDP (USD)')
plt.savefig('./Images/GDP_Vs_Time(Years).png')

#converting dataFrame to numpy arrays
arr = df.to_numpy()

#function to calculate statistics of the 3 features
def cal_stats(arr, index):
    for row in arr:
        feature = arr[:, index]
        feature_mean = round(feature.mean(),2)
        feature_var = round(feature.var(),2)
        feature_std = round(feature.std(),3)
        
    return feature_mean, feature_var, feature_std

#sales statistics
sales_stats = cal_stats(arr, 1)
print(f"The Sales mean is: {sales_stats[0]}, \
and the variance is: {sales_stats[1]}, \
with standard deviation: {sales_stats[2]}")

#adbudget statistics
adbudget_stats = cal_stats(arr, 2)
print(f"The Adbudgets mean is: {adbudget_stats[0]}, \
and the variance is: {adbudget_stats[1]}, \
with standard deviation: {adbudget_stats[2]}")

#gdp statistics
gdp_stats = cal_stats(arr, 3)
print(f"The GDPs mean is: {gdp_stats[0]}, \
and the variance is: {gdp_stats[1]}, \
with standard deviation: {gdp_stats[2]}")

print(" ")

#Calculate rolling mean and variance and plot them in a [2x1] subplot
#Add all y's from Y1 to Yt and then calculate the mean of each batch
def Cal_rolling_mean_var(df, col):
    batch = []
    computed_rolling_mean = []
    computed_rolling_var = []
    for row in df[col].values:
        batch.append(row)
        computed_rolling_mean.append(np.mean(batch))
        computed_rolling_var.append(np.var(batch))
    return computed_rolling_mean, computed_rolling_var     

def plot_rolling_stats(df, col):
    computed_rolling_mean, computed_rolling_var = Cal_rolling_mean_var(df, col)
    df['rolling_mean'] = computed_rolling_mean
    df['rolling_var'] = computed_rolling_var
    fig, axes= plt.subplots(2, figsize=(8,8))
    axes[0].plot('Date', 'rolling_mean', data=df)
    axes[0].set_title('Rolling mean over time')
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel(col)
    
    axes[1].plot('Date', 'rolling_var', data=df)
    axes[1].set_title('Rolling variance over time')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel(col)
    
    plt.subplots_adjust( bottom=0.1, top=0.95, wspace=0.3, hspace=0.5)
        
plot_rolling_stats(df, 'Sales')
plt.savefig('./Images/Sales_Rolling_mean_var.jpeg')

plot_rolling_stats(df, 'AdBudget')
plt.savefig('./Images/AdBudget_Rolling_mean_var.jpeg')

plot_rolling_stats(df, 'GDP')
plt.savefig('./Images/GDP_Rolling_mean_var.jpeg')

    
#AD Fuller Test
def ADF_Cal(x):
    result = adfuller(x)
    print("ADF Statistic: %f" %result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))
    if result[0] < result[4]["5%"]:
        print("Reject Ho, Data is stationary")
    else:
        print("Failed to reject Ho, Data is non-stationary")

#Uncomment to see individual results for other two features
sales= ADF_Cal(df['Sales'])
#adbudget= ADF_Cal(df['AdBudget'])
#gdp = ADF_Cal(df['GDP'])
print(" ")

#KPSS Test, with an alpha of 0.05, if test stats > critical value at 5%, then we rejct the Ho, data is no stationary
# if test stats < critical value at 5%, we fail to reject Ho, Data is stationary
def kpss_test(timeseries):
    print ('Results of KPSS Test:')
    kpsstest = kpss(timeseries, regression='c', nlags="auto")
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','Lags Used'])
    for key,value in kpsstest[3].items():
        kpss_output['Critical Value (%s)'%key] = value
    if kpss_output[0] > kpss_output[4]:
        print("We reject Ho, Data is non stationary")
    else:
        print("We fail to reject Ho, Data is stationary")
        
#uncomment to test the other two timeseries
kpss_sales = kpss_test(df['Sales'])
#kpss_adbudget= kpss_test(df['Sales'])
#kpss_gdp= kpss_test(df['GDP'])
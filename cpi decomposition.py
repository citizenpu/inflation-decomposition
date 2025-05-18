# -*- coding: utf-8 -*-
"""
Created on Sun May 18 00:54:03 2025

@author: citiz
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.api import VAR

data = pd.read_excel('C:/Users/citiz/Downloads/CPI decomposition.xlsx')
data=data.iloc[4:data.shape[0]-2,:]
data=data.rename(columns={"Name":"Date"})
data=data.set_index("Date")
data=data.rename(columns={'Current Prices':'Current Prices: All','Constant Prices':'Constant Prices: All'})
#name = [data.columns.values[i].partition(': ')[2] for i in range(data.shape[1])]
#name = [col.partition(':')[2] for col in data.columns]
#name1 = name[0:int(len(name)/2)] 注意如果没有int会报错，因为结果是float，不能作为index
#name=name[0:int(len(name)/2)]
data = data.reset_index() 
data_long = pd.melt(data,id_vars='Date',var_name='full_columns',value_name='Value')
data_long[['Price_Type', 'Industry']] = data_long['full_columns'].str.split(': ', n=1, expand=True)
data_long = data_long.drop(columns='full_columns')
data_final = data_long.pivot_table(index=['Date', 'Price_Type'],columns='Industry',values='Value')

#对原始数据进行处理
for i in range(len(name1)):
	data.iloc[:,i]=data.iloc[:,i].pct_change(periods=4)*100-data.iloc[:,i+len(name1)]

data.loc[data.index=="2015-03-31",:len(name1)]=100
data.loc[data.index.isin(["2015-03-31", "2015-06-30", "2015-09-30", "2015-12-31"]),:] = 100
#np.where(data.index >= '2025')[0][0]可以用来替代sum(data.index<'2025')
#注意range()data.shape[0]而不是data.shape[1]
for i in range(sum(data.index<'2016'),data.shape[0]):
	data.iloc[i,:]=data.iloc[i-4,:]*(1+data.iloc[i,:]/100)

# Import data and keep price values
data = pd.read_csv('C:/Users/Patrick/Documents/projects/InflationDecomposition/data/HFCE2000-2024.csv')
data['date'] = pd.to_datetime(data['REF_DATE'])
price_df = data.loc[data['Prices'] == 'Current prices']

# Transform data into price index for each column, apply和transform不同之处在于apply可以聚合数据，但是transform不改变原始数据结构
date_mask = price_df['REF_DATE'] == '2017-01'
price_df['price'] = price_df.groupby('Estimates')['VALUE'].transform( lambda x: x / x[date_mask].iloc[0] * 100)
prices = price_df.pivot(index= 'date', columns = 'Estimates', values = 'price')
prices = prices.dropna(axis='columns')

# prices.columns = prices.columns.values + '_p'
ln_prices = prices.apply(lambda x: np.log(x))

quantity_df = data.loc[data['Prices'] == '2017 constant prices']
date_mask = quantity_df['REF_DATE'] == '2017-01'
quantity_df['quantity'] = quantity_df.groupby('Estimates')['VALUE'].transform( lambda x: x / x[date_mask].iloc[0] * 100)
quantities = quantity_df.pivot(index= 'date', columns = 'Estimates', values = 'quantity')
quantities = quantities.dropna(axis='columns')
quantities.columns = quantities.columns.values + '_q'

ln_quantities = quantities.apply(lambda x: np.log(x))

def rolling_var_regression(df):
	resid_results = []
	window = 0
	while window+40 <= len(df):
		model_df = df.iloc[window:window+40,:]
		model = VAR(model_df)
		varmodel = model.fit(4)
		# Access the residuals from the fitted VAR model
		residuals = varmodel.resid
		resid_results.append(label_residuals(residuals))
		window = window + 1
	return resid_results


def label_residuals(df):
	std_error_residuals = df.std()

# Calculate the t-statistic for each residual by dividing the residuals by their standard error
	t_statistics = df / std_error_residuals
	threshold = 0.25

# Label residuals as "ambiguous" if their t-statistics are less than 0.25
	ambiguous_labels = np.abs(t_statistics) < threshold

	if ambiguous_labels.iloc[-1,].all():
		label = "A"
		return label  

# Check if all values in both columns are positive or negative
	if np.all(df.iloc[-1:, :].values[0] > 0):
		label = "D"
	elif np.all(df.iloc[-1:, :].values[0] < 0):
		label = "D"
		label = "S"
	return label  

counter = 0
residuals_results = []
for i in range(0,len(prices.columns)):
	rolling_df = pd.concat([ln_quantities.iloc[:,counter], ln_prices.iloc[:,counter]],axis=1 )
	residuals_results.append(rolling_var_regression(rolling_df))
	counter = counter + 1

residuals_results
label_df = pd.DataFrame(residuals_results).T
label_df.columns = ln_prices.columns
label_df.index = ln_prices.index[-len(residuals_results[0]):]
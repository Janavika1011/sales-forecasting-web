# %% [markdown]
# 
# # ARIMA and Seasonal ARIMA
# 
# 
# ## Autoregressive Integrated Moving Averages
# 
# The general process for ARIMA models is the following:
# * Visualize the Time Series Data
# * Make the time series data stationary
# * Plot the Correlation and AutoCorrelation Charts
# * Construct the ARIMA Model or Seasonal ARIMA based on the data
# * Use the model to make predictions
# 
# Let's go through these steps!

# %%
import numpy as npj
import pandas as pd

import matplotlib.pyplot as plt

# %%
df=pd.read_csv('perrin-freres-monthly-champagne-.csv')

# %%
df.head()

# %%
df.tail()

# %%
df.tail()

# %%
## Cleaning up the data
df.columns=["Month","Sales"]
df.head()

# %%
## Drop last 2 rows
df.drop(106,axis=0,inplace=True)

# %%
df.tail()

# %%
df.drop(105,axis=0,inplace=True)

# %%
df.tail()

# %%
# Convert Month into Datetime
df['Month']=pd.to_datetime(df['Month'])

# %%
df.head()

# %%
df.set_index('Month',inplace=True)

# %%
df.head()

# %%
df.describe()

# %% [markdown]
# ## Step 2: Visualize the Data

# %%
df.plot()

# %%
### Testing For Stationarity

from statsmodels.tsa.stattools import adfuller

# %%
test_result=adfuller(df['Sales'])

# %%
test_result

# %%
#Ho: It is non stationary
#H1: It is stationary

def adfuller_test(sales):
    result=adfuller(sales)
    labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations Used']
    for value,label in zip(result,labels):
        print(label+' : '+str(value) )
    if result[1] <= 0.05:
        print("strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data has no unit root and is stationary")
    else:
        print("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")
    

# %%
adfuller_test(df['Sales'])

# %% [markdown]
# ## Differencing

# %%
df['Sales First Difference'] = df['Sales'] - df['Sales'].shift(1)

# %%
df['Sales']

# %%
df['Sales'].shift(1)

# %%
adfuller_test(df['Sales First Difference'].dropna())

# %%
df['Seasonal First Difference']=df['Sales']-df['Sales'].shift(12)

# %%
df['Sales'].shift(12)

# %%
df.head(14)

# %%
## Again test dickey fuller test
adfuller_test(df['Seasonal First Difference'].dropna())

# %%
df['Seasonal First Difference'].plot()

# %% [markdown]
# ## Auto Regressive Model
# ![image.png](attachment:image.png)

# %%
from pandas.plotting import autocorrelation_plot
autocorrelation_plot(df['Sales'])
plt.show()

# %% [markdown]
# ### Final Thoughts on Autocorrelation and Partial Autocorrelation
# 
# * Identification of an AR model is often best done with the PACF.
#     * For an AR model, the theoretical PACF “shuts off” past the order of the model.  The phrase “shuts off” means that in theory the partial autocorrelations are equal to 0 beyond that point.  Put another way, the number of non-zero partial autocorrelations gives the order of the AR model.  By the “order of the model” we mean the most extreme lag of x that is used as a predictor.
#     
#     
# * Identification of an MA model is often best done with the ACF rather than the PACF.
#     * For an MA model, the theoretical PACF does not shut off, but instead tapers toward 0 in some manner.  A clearer pattern for an MA model is in the ACF.  The ACF will have non-zero autocorrelations only at lags involved in the model.
#     
#     p,d,q
#     p AR model lags
#     d differencing
#     q MA lags

# %%
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf

# %%
import statsmodels.api as sm

# %%
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(df['Seasonal First Difference'].iloc[13:],lags=40,ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(df['Seasonal First Difference'].iloc[13:],lags=40,ax=ax2)

# %%
# For non-seasonal data
#p=1, d=1, q=0 or 1
from statsmodels.tsa.arima.model import ARIMA

# %%
model=ARIMA(df['Sales'],order=(1,1,1))
model_fit=model.fit()

# %%
model_fit.summary()

# %%
df['forecast']=model_fit.predict(start=90,end=103,dynamic=True)
df[['Sales','forecast']].plot(figsize=(12,8))

# %%
import statsmodels.api as sm

# %%
model=sm.tsa.statespace.SARIMAX(df['Sales'],order=(1, 1, 1),seasonal_order=(1,1,1,12))
results=model.fit()

# %%
df['forecast']=results.predict(start=90,end=103,dynamic=True)
df[['Sales','forecast']].plot(figsize=(12,8))

# %%
from pandas.tseries.offsets import DateOffset
future_dates=[df.index[-1]+ DateOffset(months=x)for x in range(0,48)]

# %%
future_datest_df=pd.DataFrame(index=future_dates[1:],columns=df.columns)

# %%
future_datest_df.tail()

# %%
future_df=pd.concat([df,future_datest_df])

# %%
future_df

# %%
future_df['forecast'] = results.predict(start = 104, end = 175, dynamic= True)  
future_df[['Sales', 'forecast']].plot(figsize=(12, 8)) 

# %%




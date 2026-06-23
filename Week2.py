from scipy.io import loadmat
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from hurst import compute_Hc


# DATASET ACCESS
data = loadmat(r"C:\Users\rahul\Downloads\inputDataOHLCDaily_stocks_20120424.mat")

X=data['cl'][:,2]   
Y=data['cl'][:,4]  

print(np.corrcoef(X,Y))


#linear regression to find hedge ratio 

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X.reshape(-1,1),Y)
beta = model.coef_[0]
c = model.intercept_
print(f"Hedge Ratio = {beta}\nIntercept = {c}")



# to find whether give pair of stocks are cointegrated or not
#ADF test
spread = Y - beta*X - c
result = adfuller(spread)

print("ADF Statistic =", result[0])
print("p_value =", result[1])
print("Lag Used =", result[2])

if result[1]>0.05:
    print("Spread is not stationary OR given pairs are not cointegrated")
else:
    print("Spread is stationary OR given pairs are cointegrated")    

#hurst exponent
H= compute_Hc(spread)
print("Hurst Exponent =", H[0])


#Half life
delta_y = np.diff(spread)
y_lag = spread[:-1]
model.fit(y_lag.reshape(-1,1), delta_y)
lam = model.coef_[0]

half_life = -np.log(2)/lam
print("Half Life =", half_life)


#Linear mean reverting trading strategy

lookback = round(half_life)
spread_series = pd.Series(spread)
mean = spread_series.rolling(lookback).mean()
std = spread_series.rolling(lookback).std()

zscore = (spread_series - mean)/std


#ALL position information in dataframe
df = pd.DataFrame({
    'X': X,
    'Y': Y,
    'Spread': spread,
    'Mean': mean,
    'Std': std,
    'Zscore': zscore
})

df['Signal'] = 0

"""

1  -> Long Spread

-1 -> Short Spread

0  -> No Position

"""

for i in range(len(df)):

    if df['Zscore'][i] < -2:

        df.loc[i,'Signal'] = 1


    elif df['Zscore'][i] > 2:

        df.loc[i,'Signal'] = -1


    elif abs(df['Zscore'][i]) < 0.5:

        df.loc[i,'Signal'] = 0

print(f"Total Long spread Positions= ",(df['Signal']==1).sum())
print(f"Total Short spread Positions= ",(df['Signal']==-1).sum())
print(f"Total Exit Positions= ",(df['Signal']==0).sum())
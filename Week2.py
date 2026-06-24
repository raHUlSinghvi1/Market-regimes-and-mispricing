from scipy.io import loadmat
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from hurst import compute_Hc


# DATASET ACCESS
data = loadmat(r"C:\Users\rahul\Downloads\inputDataOHLCDaily_stocks_20120424.mat")

X=data['cl'][:,2]   
Y=data['cl'][:,4]  

LOG_x=np.log(X)
LOG_y=np.log(Y)

hedge_ratio=[]
spread_Rolling=[]
Log_spread_Rolling=[]
#dynamic regression for lookback window
lookback = 20
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LinearRegression

for i in range(lookback,len(X)):

    X_window=X[i-lookback:i]
    Y_window=Y[i-lookback:i]

    model=LinearRegression()

    model.fit(X_window.reshape(-1,1),Y_window)
    beta=model.coef_[0]
    c=model.intercept_
    hedge_ratio.append(beta)
    spread_today=Y[i]-beta*X[i]-c
    spread_Rolling.append(spread_today)
    log_spread_today=LOG_y[i]-beta*LOG_x[i] - c
    Log_spread_Rolling.append(log_spread_today)

# print(hedge_ratio[-10:-1])

#kalman filter for hedge ratio

from pykalman import KalmanFilter
obs_mat = np.vstack(
    [X, np.ones(len(X))]
).T[:, np.newaxis]

kf = KalmanFilter(
    n_dim_obs=1,
    n_dim_state=2,
    initial_state_mean=[0,0],
    initial_state_covariance=np.ones((2,2)),
    transition_matrices=np.eye(2),
    observation_matrices=obs_mat,
    observation_covariance=1,
    transition_covariance=0.001*np.eye(2)
)

state_means, state_covs = kf.filter(Y)
beta = state_means[:,0]
alpha = state_means[:,1]

# print(beta[-10:-1])



#comparing both beta
# import matplotlib.pyplot as plt
# plt.plot(hedge_ratio,label='Rolling')
# plt.plot(beta,label='Kalman')
# plt.legend()
# plt.show()

#ADF TESt
from statsmodels.tsa.stattools import adfuller
result = adfuller(spread_Rolling)
result_log=adfuller(Log_spread_Rolling)
print("ADF =", result[0])
print("p-value =", result[1])

print("ADF_log =", result_log[0])
print("p-value_log =", result_log[1])


if result[1] < 0.05:
    print("spread_Rolling is stationary ")
else:
    print("spread_Rolling is not stationary")

if result_log[1] < 0.05:
    print("spread_Rolling_log is stationary ")
else:
    print("Log_spread_Rolling is not stationary")

# Half life
spread_Rolling=np.array(spread_Rolling)
delta_y = np.diff(spread_Rolling)
y_lag = spread_Rolling[:-1]
model.fit(y_lag.reshape(-1,1), delta_y)
lam = model.coef_[0]
half_life = -np.log(2)/lam
print(half_life)

Log_spread_Rolling=np.array(Log_spread_Rolling)
log_delta_y=np.diff(Log_spread_Rolling)
log_y_lag=Log_spread_Rolling[:-1]
model.fit(log_y_lag.reshape(-1,1), log_delta_y)
log_lam=model.coef_[0]
log_half_life = -np.log(2)/log_lam
print(log_half_life)

#ZScore
lookback = round(half_life)
spread_Rolling = pd.Series(spread_Rolling)
mean = spread_Rolling.rolling(lookback).mean()
std = spread_Rolling.rolling(lookback).std()

zscore_Rolling = (spread_Rolling - mean)/std

lookback = round(log_half_life)
Log_spread_Rolling = pd.Series(Log_spread_Rolling)
mean =Log_spread_Rolling.rolling(lookback).mean()
std = Log_spread_Rolling.rolling(lookback).std()

zscore_Log = (Log_spread_Rolling - mean)/std


#trading signal
signal_Rolling = [0]*len(zscore_Rolling)

for i in range(len(zscore_Rolling)):
    if zscore_Rolling[i] > 2:
        signal_Rolling[i] = -1     # Short Spread
    elif zscore_Rolling[i] < -2:
        signal_Rolling[i] = 1      # Long Spread
    elif abs(zscore_Rolling[i]) < 0.5:
        signal_Rolling[i] = 0      # Exit


signal_Log=[0]*len(Log_spread_Rolling)

for i in range(len(Log_spread_Rolling)):
    if Log_spread_Rolling[i] > 2:
        signal_Log[i] = -1     # Short Spread
    elif Log_spread_Rolling[i] < -2:
        signal_Log[i] = 1      # Long Spread
    elif abs(Log_spread_Rolling[i]) < 0.5:
        signal_Log[i] = 0      # Exit



#PNL
PnL_Rolling=[]
PnL_Log=[]


for i in range(1,len(signal_Rolling)):

    daily_pnl = signal_Rolling[i-1]*(

        (Y[i]-Y[i-1])

        -

        beta[i-1]*(X[i]-X[i-1])

    )

    PnL_Rolling.append(daily_pnl)


for i in range(1,len(Log_spread_Rolling)):

    daily_pnl = Log_spread_Rolling[i-1]*(

        (Y[i]-Y[i-1])

        -

        beta[i-1]*(X[i]-X[i-1])

    )

    PnL_Log.append(daily_pnl)



PnL_Rolling = np.array(PnL_Rolling)
cum_PnL_Rolling = np.cumsum(PnL_Rolling)

PnL_Log = np.array( PnL_Log)
cum_PnL_Log = np.cumsum( PnL_Log)

print(cum_PnL_Log)
print(cum_PnL_Rolling)


#Clearly it shows that in PnL rolling no trade have been executed due to condition failed of zscore and log spread is working well 
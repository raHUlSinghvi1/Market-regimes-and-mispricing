from scipy.io import loadmat
import numpy as np
import pandas as pd


# DATASET ACCESS
data = loadmat(r"C:\Users\rahul\Downloads\inputDataOHLCDaily_stocks_20120424.mat")

# USING APPLE FOR TESTING
print(data['stocks'][0,2])
print(data['cl'].shape)


# RSI INDICATOR
delta=np.diff(data['cl'][:,2])
price=pd.Series(data['cl'][:,2])

gain = np.where(delta > 0, delta, 0)
loss = np.where(delta < 0, -delta, 0)
window=14
gain = pd.Series(gain)
loss = pd.Series(loss)

avg_gain = gain.rolling(window=14).mean()

avg_loss = loss.rolling(window=14).mean()

RS = avg_gain / avg_loss

RSI = 100 - (100/(1+RS))

# MACD INDICATOR
alpha12 = 2/(12+1)
alpha26 = 2/(26+1)
alpha9  = 2/(9+1)
prices = data['cl'][:,2]
Resistance = price.rolling(20).max()

EMA12 = [prices[0]]
EMA26 = [prices[0]]
SIGNAL=[0]
for i in range(1,len(price)):
     ema_today = alpha12*prices[i] + (1-alpha12)*EMA12[i-1]
     EMA12.append(ema_today)

for i in range(1,len(price)):
     ema_today = alpha26*prices[i] + (1-alpha26)*EMA26[i-1]
     EMA26.append(ema_today)

EMA12 = np.array(EMA12)

EMA26 = np.array(EMA26)
MACD = EMA12 - EMA26
for i in range(1,len(MACD)):
     ema_today = alpha9*MACD[i] + (1-alpha9)*SIGNAL[i-1]
     SIGNAL.append(ema_today)



buy_signal = [False]*len(MACD)

for i in range(1, len(MACD)):

    if MACD[i-1] < SIGNAL[i-1] and MACD[i] > SIGNAL[i]:
        buy_signal[i]=True


sell_signal = [False]*len(MACD)

for i in range(1, len(MACD)):

    if MACD[i-1] > SIGNAL[i-1] and MACD[i] < SIGNAL[i]:

        sell_signal[i]=True


# vOLUME INDICATOR
volume = pd.Series(data['vol'][:,2])
# last 20 days
SMA=volume.rolling(20).mean()

c = 1.3

volume_filter = [False]*len(volume)

for i in range(20, len(volume)):
    if volume[i]>c*SMA[i]:
        volume_filter[i]=True



ENTRY_POINT=[0]*len(price)
EXIT_POINT=[0]*len(price)

for i in range(len(RSI)):
    if buy_signal[i] !=0 and  price[i] > Resistance[i-1]:
        if RSI[i]<70:
            if volume_filter[i]==True:
                ENTRY_POINT[i]=1

for i in range(len(RSI)):
    if sell_signal[i] !=0:
        if RSI[i] > 30 or not volume_filter[i]:
          EXIT_POINT[i] = 1
    

print(ENTRY_POINT)





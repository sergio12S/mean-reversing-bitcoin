from sklearn import preprocessing
from backtester import Dataset, Backtester, Strategies
import pandas as pd
from sklearn.linear_model import LinearRegression


df = Dataset.get_data().set_index('Time')
df = Dataset.get_data().set_index('Time')
data = df['Close'].resample('1H').last().to_frame()


tp = [0.002, 0.003, 0.004, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1]
sl = list(map(lambda x: x/-1, tp))
seq = [1, 2, 3, 4, 5, 6]
results = []
for t in tp:
    for s in sl:
        for se in seq:

            data['signal'] = Strategies.momentum(data.Close.values, se)
            backtester = Backtester(df=data, takeProfit=t, stopLoss=s)
            data = backtester.do_backtest(exitPosition="signal",
                                          lag=None,
                                          comission=0.001, reverse=False)
            # data[['cumsum', 'Close']].plot(secondary_y=['Close'])
            res = {
                'tp': t,
                'sl': s,
                'seq': se,
                'return': data['cumsum'].values[-1]
            }
            print(res)
            results.append(dict(res))
results = pd.DataFrame(results)

data['signal'] = Strategies.momentum(data.Close.values, 5)
backtester = Backtester(df=data, takeProfit=0.02, stopLoss=-0.05)
data = backtester.do_backtest(exitPosition="signal",
                              lag=None,
                              comission=0.001, reverse=False)
data[['cumsum', 'Close']].plot(secondary_y=['Close'])

X = results[['sl', 'tp', 'seq']].values
y = results['return'].values
mm_scaler = preprocessing.MinMaxScaler()
X_train_minmax = mm_scaler.fit_transform(X)
# mm_scaler.transform(X)
reg = LinearRegression().fit(X_train_minmax, y)
reg.score(X_train_minmax, y)
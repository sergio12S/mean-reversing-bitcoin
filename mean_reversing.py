import requests
import pandas as pd
import numpy as np
from backtester import Backtester
from tqdm import tqdm
import pickle
# from statsmodels.tsa.stattools import adfuller
# from sklearn.tree import DecisionTreeRegressor


PARAMS = {"backtest": "all"}
df = requests.get(
    "https://aipricepatterns.com/api/api/backtest", params=PARAMS)
data = df.json()['backtest']
data = pd.DataFrame(data)
data = data.set_index('Time')
data.index = pd.to_datetime(data.index)
data['hours'] = data.index.hour
data['minutes'] = data.index.minute
data['dayofweek'] = data.index.dayofweek
# data['stationarity'] = data['Close'].rolling(
#     72).apply(lambda x: adfuller(x)[1])


SIGMA = np.arange(1, 4, 1)
LAG = np.arange(1, 10, 1)
WINDOW_MA = np.arange(50, 200, 50)
WINDOW_STD = np.arange(100, 500, 100)  # np.arange(100, 500, 50)


def optimal_parameter(sigma, lag, window_ma, window_std, plot=False):

    # 1 Describe value for our strategy
    data['chg'] = data['Close'].pct_change(1)
    data['ma_24'] = data['chg'].rolling(window=window_ma).median()
    data['std_ma_24'] = data['ma_24'].rolling(window=window_std).std()

    # 2 Predictible value. Some logic
    data['x1'] = np.where(
        (data['chg'] > data['std_ma_24'] * sigma)
        & (data['chg'] > 0), data['std_ma_24'], data['chg'])

    data['x1'] = np.where(
        (data['chg'] < -data['std_ma_24'] * sigma)
        & (data['chg'] < 0), -data['std_ma_24'], data['x1']
    )

    # 3 Create signal
    data['signal'] = np.where(
        (data['x1'] == -data['std_ma_24']) &
        (data['x1'] < 0), 1, 0
    )
    data['signal'] = np.where(
        (data['x1'] == data['std_ma_24']) &
        (data['x1'] > 0), -1, data['signal']
    )

    #  4 Test. Find best parameters.
    data['lag'] = data['Close'].shift(-lag).pct_change()
    if plot:
        data[data['signal'] == -1]['lag'].mean()
        data[data['signal'] == 1]['lag'].cumsum().plot()
        data[data['signal'] == -1]['lag'].cumsum().plot()
    return data


def create_rules(data, sigma, lag, window_ma, window_std, rules=10):
    back = Backtester(df=data, takeProfit=0.005, stopLoss=-0.005)
    count_rules = 0

    data_rules = []
    for s in tqdm(SIGMA):
        for la in LAG:
            for w_ma in WINDOW_MA:
                for w_std in WINDOW_STD:
                    data = optimal_parameter(
                        sigma=s, lag=la, window_ma=w_ma, window_std=w_std,
                        plot=False
                    )
                    data = back.do_backtest(exitPosition="signal",
                                            lag=LAG,
                                            comission=0,
                                            reverse=False)

                    # How works statregy depends on day of week
                    analyse_data_of_week = data[['return', 'dayofweek']]\
                        .groupby('dayofweek').agg(['mean']).reset_index()
                    analyse_data_of_week.columns = ['day_of_week', 'mean']

                    # How works statregy depends on hours
                    analyse_hours = data[['return', 'hours']].\
                        groupby('hours').agg(['mean']).reset_index()
                    analyse_hours.columns = ['hours', 'mean']

                    # How works statregy depends on minutes
                    analyse_minutes = data[['return', 'minutes']].\
                        groupby('minutes').agg(['mean']).reset_index()
                    analyse_minutes.columns = ['minutes', 'mean']

                    statistics = {
                        "cumsum": data['cumsum'].values[-1],
                        "mean":  data[data['return'] != 0]['return'].mean(),
                        "median": data[data['return'] != 0]['return'].median(),
                        'count': data[data['return'] != 0]['return'].count(),
                        'rules': {
                            "sigma": s,
                            "lag": la,
                            "window_ma": w_ma,
                            "window_std": w_std,
                            "day_of_week": analyse_data_of_week
                            .to_dict(orient="records"),
                            "hours": analyse_hours
                            .to_dict(orient="records"),
                            "minutes": analyse_minutes
                            .to_dict(orient="records")
                        }
                    }
                    # print(statistics)
                    data_rules.append(dict(statistics))
                    count_rules += 1
                    print('Rules: ', count_rules)
                    print('Backtest result: ', data['cumsum'].values[-1])
                    if count_rules == rules:
                        return data_rules
    return data_rules


data_rules = create_rules(
    data=data,
    sigma=SIGMA,
    lag=LAG,
    window_ma=WINDOW_MA,
    window_std=WINDOW_STD,
    rules=300
)
# Save rules
with open("rules_strategy.pickle", 'wb') as p:
    pickle.dump(data_rules, p, protocol=pickle.HIGHEST_PROTOCOL)

# Load rules
with open("rules_strategy.pickle", 'rb') as p:
    data_rules = pickle.load(p)

# Top N rules
TOP = 1
best_rules = sorted(data_rules, key=lambda i: i['cumsum'], reverse=True)[
    :TOP][0]['rules']
best_rules['minutes']

# Best strategy
data = optimal_parameter(sigma=best_rules['sigma'], lag=best_rules['lag'],
                         window_ma=best_rules['window_ma'],
                         window_std=best_rules['window_std'],
                         plot=False)

# Time Rules for Mean Revesing strategy

# 1 Exclude day 6
data['signal'] = np.where(data['dayofweek'] == 6, 0, data['signal'])

# 2 Exclude hours
data['signal'] = np.where(data['hours'] == 0, 0, data['signal'])
data['signal'] = np.where(data['hours'] == 15, 0, data['signal'])
data['signal'] = np.where(data['hours'] == 14, 0, data['signal'])
data['signal'] = np.where(data['hours'] == 21, 0, data['signal'])
data['signal'] = np.where(data['hours'] == 22, 0, data['signal'])


# 3 Delete stationarity HM
# https://www.tradelikeamachine.com/blog/cointegration-pairs-trading/part-3-stationary-time-series-for-pairs-trading-systems
data['signal'] = np.where((data['stationarity'] >= 0.2) & (
    data['stationarity'] <= 0.4), data['signal'], 0)


# 4 Apply machine learning
loaded_model = pickle.load(open('ml_model_tree', 'rb'))
WINDOW_MA = np.arange(50, 200, 10)
# Define new WINDOW_MA variable
for i in WINDOW_MA:
    data['chg'] = data['Close'].pct_change(1)
    data['ma_24'] = data['chg'].rolling(window=i).median()
    data[f'window_{i}'] = data['ma_24'].rolling(window=i).std()
data = data.dropna()
columns = ['Volume',
           'hours',
           'minutes',
           'dayofweek',
           'chg',
           'ma_24',
           'window_50',
           'window_60',
           'window_70',
           'window_80',
           'window_90',
           'window_100',
           'window_110',
           'window_120',
           'window_130',
           'window_140',
           'window_150',
           'window_160',
           'window_170',
           'window_180',
           'window_190']
data['predict'] = loaded_model.predict(
    data.loc[:, columns].values)

# HM change predict value from 0 to 0.005
data['signal'] = np.where(
    ((data['predict'] > 0.005) & (data['signal'] == 1)) |
    ((data['predict'] < -0.005) & (data['signal'] == -1)),
    data['signal'], 0)


# Try backtest
back = Backtester(df=data, takeProfit=0.005, stopLoss=-0.005)
data = back.do_backtest(exitPosition="signal",
                        lag=best_rules['lag'],
                        comission=0,
                        reverse=False)

data['cumsum'].plot()

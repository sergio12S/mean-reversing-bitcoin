import requests
import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeRegressor  # the dt regressor
from sklearn.model_selection import train_test_split
import pickle
# import matplotlib.pyplot as plt
# from sklearn import tree
# import graphviz


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


WINDOW_MA = np.arange(50, 200, 10)
# HM
LAG = 3  # we could change
# Define new WINDOW_MA variable
for i in WINDOW_MA:
    data['chg'] = data['Close'].pct_change(1)
    data['ma_24'] = data['chg'].rolling(window=i).median()
    data[f'window_{i}'] = data['ma_24'].rolling(window=i).std()

data['y'] = data['Close'].shift(-LAG).pct_change()
data = data.dropna()


# Prepare sataset
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
X = data.loc[:, columns].values
y = data['y'].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1)

# Train model
dregressor = DecisionTreeRegressor()
dregressor = dregressor.fit(X_train, y_train)
pickle.dump(dregressor, open('ml_model_tree', 'wb'))

# Analyse accuracy
# predict = np.where(dregressor.predict(X_test) > 0, 1, 0)
# fact = np.where(y_test > 0, 1, 0)
# np.sum(predict == fact) / len(X_test)
# plt.scatter(dregressor.predict(X_train), y_train)
# dot_data = tree.export_graphviz(dregressor, feature_names=list(
#     data.loc[:, 'window_50':'window_190'].columns), filled=True)
# graph = graphviz.Source(dot_data, format='png')
# graph

# if __name__ == "__main__":
#     pass

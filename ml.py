import requests
import pandas as pd
import numpy as np
from backtester import Backtester
from tqdm import tqdm
import pickle
from sklearn.tree import DecisionTreeRegressor  # the dt regressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import tree
import graphviz
from six import StringIO
from IPython.display import Image
import pydotplus
from IPython.core.pylabtools import figsize


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

for i in WINDOW_MA:
    print(i)
    data['chg'] = data['Close'].pct_change(1)
    data['ma_24'] = data['chg'].rolling(window=i).median()
    data[f'window_{i}'] = data['ma_24'].rolling(window=i).std()

data['y'] = data['Close'].shift(-1).pct_change()
data = data.dropna()


# Prepare sataset
X = data.loc[:, 'window_50':'window_190'].values
y = data['y'].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1)

# Train model
dregressor = DecisionTreeRegressor()
dregressor = dregressor.fit(X_train, y_train)

plt.scatter(dregressor.predict(X_train), y_train)
dot_data = tree.export_graphviz(dregressor, feature_names=list(
    data.loc[:, 'window_50':'window_190'].columns), filled=True)
graph = graphviz.Source(dot_data, format='png')
graph

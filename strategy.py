import requests
import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeRegressor  # the dt regressor
import pickle
from backtester import Backtester
from tqdm import tqdm
# import matplotlib.


class MeanReversing:
    def __init__(self):
        self.WINDOW_MA = np.arange(50, 200, 10)
        self.columns = ['Volume',
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
        self.SIGMA = np.arange(1, 4, 1)
        self.LAG = np.arange(1, 10, 1)
        # self.WINDOW_MA = np.arange(50, 200, 50)
        self.WINDOW_STD = np.arange(100, 500, 100)
        self.rules = []

    def _get_dataset(self):

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
        return data

    def _create_features(self, data):
        for i in self.WINDOW_MA:
            data['chg'] = data['Close'].pct_change(1)
            data['ma_24'] = data['chg'].rolling(window=i).median()
            data[f'window_{i}'] = data['ma_24'].rolling(window=i).std()
        return data

    def _machine_learning_tree(self, data):
        X = data.loc[:, self.columns].values
        y = data['y'].values
        dregressor = DecisionTreeRegressor()
        dregressor = dregressor.fit(X, y)
        pickle.dump(dregressor, open('ml_model_tree', 'wb'))

    def _optimal_parameter(self, sigma, lag, window_ma,
                           window_std, plot=False):

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

    def _create_rules(self, data, sigma, lag, window_ma, window_std, rules=10):
        back = Backtester(df=data, takeProfit=0.005, stopLoss=-0.005)
        count_rules = 0

        data_rules = []
        for s in tqdm(self.SIGMA):
            for la in self.LAG:
                for w_ma in self.WINDOW_MA:
                    for w_std in self.WINDOW_STD:
                        data = self.optimal_parameter(
                            sigma=s, lag=la, window_ma=w_ma, window_std=w_std,
                            plot=False
                        )
                        data = back.do_backtest(exitPosition="signal",
                                                lag=self.LAG,
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
                            "mean":
                            data[data['return'] != 0]['return'].mean(),
                            "median":
                            data[data['return'] != 0]['return'].median(),
                            'count':
                            data[data['return'] != 0]['return'].count(),
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

    def _save_rules(self, rules=300):
        self.rules = self.create_rules(
            data=data,
            sigma=self.SIGMA,
            lag=self.LAG,
            window_ma=self.WINDOW_MA,
            window_std=self.WINDOW_STD,
            rules=rules
        )
        # Save rules
        with open("rules_strategy.pickle", 'wb') as p:
            pickle.dump(self.rules, p, protocol=pickle.HIGHEST_PROTOCOL)

    def _load_rules(self):
        with open("rules_strategy.pickle", 'rb') as p:
            self.rules = pickle.load(p)

    def create_machine_learning_models(self):
        data = self._get_dataset()
        data = self._create_features(data)
        self._machine_learning_tree()

    def _best_rules(self):
        self._load_rules()
        # Top N rules
        TOP = 1
        best_rules = sorted(self.rules, key=lambda i: i['cumsum'],
                            reverse=True)[
            :TOP][0]['rules']
        best_rules['minutes']
        return

    def create_signal(self):
        data = self._get_dataset()
        data = self._create_features(data)
        loaded_model = pickle.load(open('ml_model_tree', 'rb'))
        best_rules = self._best_rules()
        data = self._optimal_parameter(sigma=best_rules['sigma'],
                                       lag=best_rules['lag'],
                                       window_ma=best_rules['window_ma'],
                                       window_std=best_rules['window_std'],
                                       plot=False)
        data['predict'] = loaded_model.predict(
            data.loc[:, self.columns].values)
        data['signal'] = np.where(
            ((data['predict'] > 0) & (data['signal'] == 1)) |
            ((data['predict'] < 0) & (data['signal'] == -1)),
            data['signal'], 0)
        return data


strategy = MeanReversing()
data = strategy.create_signal()

# Debug our code
data = strategy._get_dataset()
data = strategy._create_features(data)
loaded_model = pickle.load(open('ml_model_tree', 'rb'))
best_rules = strategy._best_rules()

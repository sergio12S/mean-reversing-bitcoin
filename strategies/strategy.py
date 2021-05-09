from loguru import logger
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import pickle
from datetime import datetime
from backtester import Backtest
from manager import Manager
from scipy import stats
from statsmodels.tsa.stattools import grangercausalitytests
from itertools import count

logger.add('debug.json', format='{time} {level} {message}',
           level='WARNING', rotation='6:00',
           compression='zip', serialize=True)


class StatisticalTest:
    @staticmethod
    def kstest(data):
        '''
        The Kolmogorov-Smirnov test can evaluate whether
        a distribution is normal.
        It provides us with a p-value.
        If this value is significant (< 0.05),
        sthen the data is not normal:
        '''
        test = stats.kstest(data, cdf="norm")
        if test.pvalue <= 0.05:
            '''
            not random
            '''
            return True
        else:
            return False

    @staticmethod
    def t_test(x, y):
        '''
        What is  T - test?
        https://www.investopedia.com/terms/t/t-test.asp
        With this test, we determine that the values are not random.
        If the values are greater than 5%,
        then we can reject the hypothesis that the values are random.
        '''
        test = stats.ttest_ind(
            x,
            y, equal_var=False
        )
        if test.pvalue >= 0.05:
            '''
            not random
            '''
            return True
        else:
            return False

    @staticmethod
    def grange(data, lag):
        return grangercausalitytests(data, lag)


# Monitoring strategy

class StrategiesSignals:
    @staticmethod
    def mean_reversing_sigma(data, sigma, lag, window_ma,
                             window_std, plot=False):
        temp = data.copy()

        # 1 Describe value for our strategy
        temp.loc[:, 'chg'] = temp['close'].pct_change(1)
        temp.loc[:, 'ma_24'] = temp['chg'].rolling(window=window_ma).median()
        temp.loc[:, 'std_ma_24'] = temp['ma_24'].rolling(
            window=window_std).std()

        # 2 Predictible value. Some logic
        temp.loc[:, 'x1'] = np.where(
            (temp['chg'] > temp['std_ma_24'] * sigma)
            & (temp['chg'] > 0), temp['std_ma_24'], temp['chg'])

        temp.loc[:, 'x1'] = np.where(
            (temp['chg'] < -temp['std_ma_24'] * sigma)
            & (temp['chg'] < 0), -temp['std_ma_24'], temp['x1']
        )

        # 3 Create signal
        temp.loc[:, 'signal'] = np.where(
            (temp['x1'] == -temp['std_ma_24']) &
            (temp['x1'] < 0), 1, 0
        )
        temp.loc[:, 'signal'] = np.where(
            (temp['x1'] == temp['std_ma_24']) &
            (temp['x1'] > 0), -1, temp['signal']
        )

        #  4 Test. Find best parameters.
        if plot:
            temp.loc[:, 'lag'] = temp['close'].shift(-lag).pct_change()
            temp[temp['signal'] == -1]['lag'].mean()
            temp[temp['signal'] == 1]['lag'].cumsum().plot()
            temp[temp['signal'] == -1]['lag'].cumsum().plot()
        return temp

    @staticmethod
    def mean_reversing(data, window):
        data['signal_mean_reversing'] = 0
        data = data.reset_index()
        data['reversion'] = data['close'].pct_change(1)
        for i, v in data.iterrows():
            if i < window:
                continue
            temp = data[i-window: i]
            resistance = temp[temp['reversion'] > 0]['reversion'].mean()
            support = temp[temp['reversion'] < 0]['reversion'].mean()
            if v['reversion'] < support:
                data.loc[i, 'signal_mean_reversing'] = 1
            if v['reversion'] > resistance:
                data.loc[i, 'signal_mean_reversing'] = -1
        data = data.set_index('time')
        return data['signal_mean_reversing'].values

    @staticmethod
    def momentum(df, seq):
        signals = np.zeros(len(df))
        cons_day = 0
        for k in range(1, len(df)):
            price = df[k]
            prior_price = df[k-1]
            if price > prior_price:
                if cons_day < 0:
                    cons_day = 0
                cons_day += 1
            if price < prior_price:
                if cons_day > 0:
                    cons_day = 0
                cons_day -= 1
            if cons_day == seq:
                signals[k] = 1
            if cons_day == -seq:
                signals[k] = -1
        return signals

    @staticmethod
    def unusual_volume(df, seq):
        # Trade when we have 5% of trading voluem in 1H
        pass


class GridSearch:
    def __init__(self, data, money_managment):
        self.data = data
        self.money_managment = money_managment
        self.window_ma = np.arange(50, 200, 10)
        self.window_std = np.arange(100, 500, 100)
        self.lag = np.arange(1, 10, 1)
        self.sigma = np.arange(1, 4, 1)

    def search(self, data, rules=300):

        counter = count(0)
        rules = [
            self.statistics(
                lag=la,
                sigma=s,
                window_ma=w_ma,
                window_std=w_std
            ) for la in self.lag
            for s in self.sigma
            for w_ma in self.window_ma
            for w_std in self.window_std
            if next(counter) < rules
        ]
        return rules

    def statistics(self, lag, sigma, window_ma, window_std):
        temp = StrategiesSignals.mean_reversing_sigma(
            data=self.data,
            sigma=sigma, lag=lag,
            window_ma=window_ma,
            window_std=window_std,
            plot=False
        )
        back = Backtest()
        temp = back.exit_by_signal(
            data=temp,
            take_profit=self.money_managment.get(
                "take_profit"),
            stop_loss=self.money_managment.get("stop_loss"),
            comission=0
        )

        # How works statregy depends on day of week
        analyse_data_of_week = temp[['return', 'dayofweek']]\
            .groupby('dayofweek').agg(['mean']).reset_index()
        analyse_data_of_week.columns = ['day_of_week', 'mean']

        # How works statregy depends on hours
        analyse_hours = temp[['return', 'hours']].\
            groupby('hours').agg(['mean']).reset_index()
        analyse_hours.columns = ['hours', 'mean']

        # How works statregy depends on minutes
        analyse_minutes = temp[['return', 'minutes']].\
            groupby('minutes').agg(['mean']).reset_index()
        analyse_minutes.columns = ['minutes', 'mean']

        statistics = {
            "cumsum": temp['cumsum'].values[-1],
            "mean":
            temp.loc[temp['return'] != 0, 'return'].mean(),
            "median":
            temp.loc[temp['return'] != 0, 'return'].median(),
            'count':
                temp.loc[temp['return'] != 0, 'return'].count(),
                'rules': {
                    "sigma": sigma,
                    "lag": lag,
                    "window_ma": window_ma,
                    "window_std": window_std,
                    "day_of_week": analyse_data_of_week
                    .to_dict(orient="records"),
                    "hours": analyse_hours
                    .to_dict(orient="records"),
                    "minutes": analyse_minutes
                    .to_dict(orient="records")
            }
        }
        print(statistics)
        return statistics


class MlDt:
    def __init__(self, threshold=0.003,
                 name_strategy='ML DT',
                 ticker='SFPUSDT', size=10):
        '''
        threshold - parameter for otimize signal.
        If threshold = false. Strategy not use ml.

        '''
        self.ticker = ticker
        self.size = size
        self.mean_reversing_window = [25, 50, 75]
        self.momentum_seq = np.arange(1, 10, 1)
        self.window_ma_ml = np.arange(10, 100, 10)
        self.columns = []
        self.rules = []
        self.threshold = threshold
        self.money_managment = {
            "stop_loss": -0.05,
            "take_profit": 0.05,
            "exit_position": 'take',
            "lag": 12
        }
        self.name_strategy = name_strategy

    @ logger.catch
    def _create_features(self, data):
        temp = data.copy()
        temp.loc[:, 'hours'] = temp.index.hour
        temp.loc[:, 'minutes'] = temp.index.minute
        temp.loc[:, 'dayofweek'] = temp.index.dayofweek
        temp.loc[:, 'unusual_volume'] = pd.qcut(
            data['volume'], 5,  labels=[1, 2, 3, 4, 5])

        self.columns = [
            'unusual_volume',
            'hours',
            'minutes',
            'dayofweek',
            'chg',
            'ma_24'
        ]
        temp.loc[:, 'chg'] = temp['close'].pct_change(1)
        for i in self.window_ma_ml:

            temp.loc[:, 'ma_24'] = temp['chg'].rolling(window=i).median()
            temp.loc[:, f'window_{i}'] = temp['ma_24'].rolling(window=i).std()
            self.columns.append(f'window_{i}')

        # Create mean reversing signals
        for i in self.mean_reversing_window:
            temp.loc[:, f'signal_mean_reversing_{i}'] = \
                StrategiesSignals.mean_reversing(
                    temp, window=i
            )
            self.columns.append(f'signal_mean_reversing_{i}')

        # Create momentum signals
        for i in self.momentum_seq:
            temp.loc[:, f'signal_momentum_{i}'] = \
                StrategiesSignals.momentum(
                    temp['close'].values, seq=i
            )
            self.columns.append(f'signal_momentum_{i}')

        return temp

    @ logger.catch
    def _machine_learning_tree(self, data):
        '''
        Optimizing this strategy with machine learning
        '''
        temp = data.copy()
        temp = temp.loc[:,  self.columns + ['y']].dropna()
        X = temp.loc[:, self.columns].dropna().values
        y = temp.loc[:, 'y'].values
        dregressor = DecisionTreeRegressor()
        dregressor = dregressor.fit(X, y)
        pickle.dump(dregressor, open(
            'ml_model_tree', 'wb'))
        return

    def _create_rules(self, data, rules=300):
        temp = data.copy()
        temp = self._create_features(temp).dropna()
        grid = GridSearch(data=temp, money_managment=self.money_managment)
        return grid.search(data=temp, rules=rules)

    def _save_rules(self, data, rules=300):
        self.rules = self._create_rules(
            data=data,
            rules=rules
        )
        # Save rules
        if self.rules:
            with open(
                'rules_strategy.pickle',
                    'wb') as p:
                pickle.dump(self.rules, p, protocol=pickle.HIGHEST_PROTOCOL)

    def _load_rules(self):
        with open(
            'rules_strategy.pickle',
                'rb') as p:
            self.rules = pickle.load(p)

    def _best_rules(self):
        self._load_rules()
        # Top N rules
        TOP = 1
        best_rules = sorted(self.rules, key=lambda i: i['cumsum'],
                            reverse=True)[
            :TOP][0]['rules']
        best_rules['minutes']
        return best_rules

    @ logger.catch
    def create_machine_learning_models(self, data, split_train=1, lag=1):
        temp = data.copy()
        temp.loc[:, 'y'] = temp['close'].shift(-lag).pct_change()
        if split_train == 1:
            temp = self._create_features(temp)
            self._machine_learning_tree(data=temp)
            temp = self.backtest(temp,
                                 comission=0.001,
                                 takeProfit=self.money_managment.get(
                                     "take_profit"),
                                 stopLoss=self.money_managment.get(
                                     "stop_loss"),
                                 exitPosition=self.money_managment.get(
                                     "exit_position"),
                                 ml=True)
            temp['cumsum'].plot()
        if split_train < 1:
            temp = temp.reset_index()
            train = temp.loc[:int(len(temp) * split_train), :]
            test = temp.loc[int(len(temp) * split_train):, :]
            train = train.set_index('time')
            test = test.set_index('time')
            train = self._create_features(train)
            self._machine_learning_tree(data=train)

            train = self.backtest(train,
                                  comission=0.001,
                                  takeProfit=self.money_managment.get(
                                      "take_profit"),
                                  stopLoss=self.money_managment.get(
                                      "stop_loss"),
                                  exitPosition=self.money_managment.get(
                                      "exit_position"),
                                  ml=True)
            test = self.backtest(test,
                                 comission=0.001,
                                 takeProfit=self.money_managment.get(
                                     "take_profit"),
                                 stopLoss=self.money_managment.get(
                                     "stop_loss"),
                                 exitPosition=self.money_managment.get(
                                     "exit_position"),
                                 ml=True)
            train['cumsum'].plot()
            test['cumsum'].plot()

    @ logger.catch
    def do_signals(self, data):
        temp = data.copy()
        temp = self._create_features(temp)
        loaded_model = pickle.load(
            open('ml_model_tree', 'rb'))
        best_rules = self._best_rules()
        temp = StrategiesSignals\
            .mean_reversing_sigma(data=temp,
                                  sigma=best_rules['sigma'],
                                  lag=best_rules['lag'],
                                  window_ma=best_rules['window_ma'],
                                  window_std=best_rules['window_std'],
                                  plot=False)

        temp = temp.dropna()
        temp.loc[:, 'predict'] = loaded_model.predict(
            temp.loc[:, self.columns].values)
        if self.threshold:
            # Use machine learning to optimize model
            temp.loc[:, 'signal'] = np.where(
                ((temp['predict'] > self.threshold) & (temp['signal'] == 1)) |
                ((temp['predict'] < -self.threshold) & (temp['signal'] == -1)),
                temp['signal'], 0)

        return temp.tail(1)

    @ logger.catch
    def _open_position(self, data, session, logic, signal):
        curent_price = float(data['close'].iloc[-1])
        if signal == 1:
            new_trade = logic(
                Strategy="ML DT",
                Status="open",
                Time=data.index[-1].to_pydatetime(),
                Open=curent_price,
                Lag=1,
                Signal=signal
            )
            session.add(new_trade)
            session.commit()
        # Sell
        if signal == -1:
            new_trade = logic(
                Strategy="ML DT",
                Status="open",
                Time=datetime.utcnow().isoformat(),
                Open=curent_price,
                Lag=1,
                Signal=signal
            )
            session.add(new_trade)
            session.commit()

    @ logger.catch
    def _check_open_position(self, session, logic):
        events = session.query(logic).with_entities(
            logic.Strategy,
            logic.Status,
            logic.Time,
            logic.Open,
            logic.Close,
            logic.Lag,
            logic.Signal,
            logic.Rule,
            logic.Result,
            logic.Cumsum
        ).filter(
            logic.Strategy == "ML DT",
            logic.Status == "open"
        ).order_by(logic.Time.desc()).first()
        return events

    @ logger.catch
    def _close_position(self, session, logic, current_price, events):
        profit = (current_price / events.Open - 1) / events.Signal
        session.query(logic).filter(
            logic.Status == "open",
            logic.Strategy == "ML DT"
        ).update(
            {
                "Close": float(current_price),
                "Status": "close",
                "Result": float(profit)
            }, synchronize_session=False
        )
        session.commit()
        send_status = {
            'name': 'ML DT',
            'status': 'close',
            'time': events.Time,
            'Close': current_price,
            'signal': events.Signal,
            'profit': profit
        }
        return send_status

    @ logger.catch
    def run_strategy(self, data):
        signal = self.do_signals(data)
        manager = Manager(name_strategy="ML DT",
                          data=signal,
                          take_profit=self.money_managment.get("take_profit"),
                          stop_loss=self.money_managment.get("stop_loss"),
                          lag=self.money_managment.get("lag"),
                          ticker=self.ticker,
                          size=self.size
                          )
        status = manager.manage()
        return status

    @ logger.catch
    def backtest(self,
                 data,
                 lag=1,
                 comission=0,
                 exitPosition="signal",
                 takeProfit=0.005,
                 stopLoss=-0.005,
                 ml=True):
        temp = data.copy()
        temp = self._create_features(temp)
        loaded_model = pickle.load(
            open('ml_model_tree', 'rb'))
        # best_rules = self._best_rules()
        # temp = StrategiesSignals\
        #     .mean_reversing_sigma(data=temp,
        #                           sigma=best_rules['sigma'],
        #                           lag=best_rules['lag'],
        #                           window_ma=best_rules['window_ma'],
        #                           window_std=best_rules['window_std'],
        #                           plot=False)
        temp['signal'] = StrategiesSignals\
            .mean_reversing(data=temp, window=100)
        temp = temp.dropna()
        if ml:
            if not self.threshold:
                logger.info(
                    'threshold shoud be float value for machine learning mode')

            temp.loc[:, 'predict'] = loaded_model.predict(
                temp[self.columns].values)
            temp.loc[:, 'signal'] = np.where(
                ((temp['predict'] > self.threshold) & (temp['signal'] == 1)) |
                ((temp['predict'] < -self.threshold) & (temp['signal'] == -1)),
                temp['signal'], 0)

        back = Backtest()
        if exitPosition == 'signal':
            temp = back.exit_by_signal(
                data=temp,
                take_profit=takeProfit,
                stop_loss=stopLoss,
                comission=comission
            )
            logger.info('Backtest using signal method')
            return temp
        if exitPosition == 'take':
            temp = back.exit_by_take(
                data=temp,
                take_profit=takeProfit,
                stop_loss=stopLoss,
                comission=comission
            )
            logger.info('Backtest using take profit method')
            return temp
        if exitPosition == 'lag':
            temp = back.exit_by_lag(
                data=temp,
                take_profit=takeProfit,
                stop_loss=stopLoss,
                lag=lag,
                comission=comission
            )
            logger.info('Backtest using lag method')
            return temp

import numpy as np


class Backtester:
    def __init__(self, df, takeProfit, stopLoss):
        self.df = df
        self.takeProfit = takeProfit
        self.stopLoss = stopLoss
        self.positionData = {}
        self.status = 'close'
        self.countLag = 0

    def _check_mm(self, position, change_price):
        if position == "buy":
            check_mm = (change_price >= self.takeProfit) or (
                change_price <= self.stopLoss)
            return check_mm
        if position == "sell":
            change_price = change_price / -1
            check_mm = (change_price >= self.takeProfit) or (
                change_price <= self.stopLoss)
            return check_mm
        else:
            return False

    def _check_signal(self, position):
        if self.positionData == {}:
            return False
        if position == "buy":
            check_signal = self.positionData['signal'] == 1
            return check_signal
        if position == 'sell':
            check_signal = self.positionData['signal'] == -1
            return check_signal
        else:
            return False

    def _check_status(self):
        if self.status == 'open':
            return True
        else:
            return False

    def _close_position(self, position, iterator):
        if self.positionData == {}:
            return False
        if position == "buy":
            check_position = self.positionData['signal'] == 1 and\
                iterator['signal'] == -1 and \
                self.status == 'open'
            return check_position
        if position == "sell":
            check_position = self.positionData['signal'] == -1 and\
                iterator['signal'] == 1 and \
                self.status == 'open'
            return check_position

    def _close_position_lag(self, position, lag, iterator):
        if self.positionData == {}:
            return False
        if position == "buy":
            check_position = self.positionData['signal'] == 1 and\
                self.countLag == lag and \
                self.status == 'open'
            return check_position
        if position == "sell":
            check_position = self.positionData['signal'] == -1 and\
                self.countLag == lag and \
                self.status == 'open'
            return check_position

    def _open_position(self, position, iterator):
        if position == "buy":
            check_signal = self.status == 'close' and iterator['signal'] == 1
            return check_signal

        if position == "sell":
            check_signal = self.status == 'close' and iterator['signal'] == -1
            return check_signal

    def exit_by_signal(self):
        for i, v in self.df.iterrows():
            if len(self.positionData):
                change_price = (v['Close'] / self.positionData['Close']) - 1
                # money managment
                # long position
                if self._check_mm(position="buy",
                                  change_price=change_price) and\
                        self._check_signal(position='buy') and\
                        self._check_status():
                    self.df.loc[i, 'return'] = change_price
                    self.positionData = {}
                    self.status = 'close'
                # short position
                if self._check_mm(position="sell",
                                  change_price=change_price) and\
                        self._check_signal(position='sell') and\
                        self._check_status():
                    self.df.loc[i, 'return'] = change_price / -1
                    self.positionData = {}
                    self.status = 'close'
                # close open long
                if self._close_position(position="buy", iterator=v):
                    self.df.loc[i, 'return'] = change_price
                    self.positionData = {}
                    self.status = 'close'
                # close open short
                if self._close_position(position="sell", iterator=v):
                    self.df.loc[i, 'return'] = change_price / -1
                    self.positionData = {}
                    self.status = 'close'
            # open long position
            if self._open_position(position='buy', iterator=v):
                self.positionData = v.to_dict()
                self.status = 'open'
            # open short position
            if self._open_position(position='sell', iterator=v):
                self.positionData = v.to_dict()
                self.status = 'open'

    def exit_by_take(self):
        for i, v in self.df.iterrows():
            if len(self.positionData):
                change_price = (v['Close'] / self.positionData['Close']) - 1
                # money managment
                # long position
                if self._check_mm(position="buy",
                                  change_price=change_price) and\
                        self._check_signal(position='buy') and\
                        self._check_status():
                    self.df.loc[i, 'return'] = change_price
                    self.positionData = {}
                    self.status = 'close'
                # short position
                if self._check_mm(position="sell",
                                  change_price=change_price) and\
                        self._check_signal(position='sell') and\
                        self._check_status():
                    self.df.loc[i, 'return'] = change_price / -1
                    self.positionData = {}
                    self.status = 'close'
            # open long position
            if self._open_position(position='buy', iterator=v):
                self.positionData = v.to_dict()
                self.status = 'open'
            # open short position
            if self._open_position(position='sell', iterator=v):
                self.positionData = v.to_dict()
                self.status = 'open'

    def exit_by_lag(self, lag):
        for i, v in self.df.iterrows():
            if len(self.positionData):
                change_price = (v['Close'] / self.positionData['Close']) - 1
                # money managment
                # long position
                if self._check_mm(position="buy",
                                  change_price=change_price) and\
                        self._check_signal(position='buy') and\
                        self._check_status():
                    self.df.loc[i, 'return'] = change_price
                    self.positionData = {}
                    self.status = 'close'
                # short position
                if self._check_mm(position="sell",
                                  change_price=change_price) and\
                        self._check_signal(position='sell') and\
                        self._check_status():
                    self.df.loc[i, 'return'] = change_price / -1
                    self.positionData = {}
                    self.status = 'close'
                # close open long
                if self._close_position_lag(position="buy",
                                            lag=lag,
                                            iterator=v):
                    self.df.loc[i, 'return'] = change_price
                    self.positionData = {}
                    self.status = 'close'
                    self.countLag = 0
                # close open short
                if self._close_position_lag(position="sell",
                                            lag=lag, iterator=v):
                    self.df.loc[i, 'return'] = change_price / -1
                    self.positionData = {}
                    self.status = 'close'
                    self.countLag = 0
            # open long position
            if self._open_position(position='buy', iterator=v):
                self.positionData = v.to_dict()
                self.status = 'open'
            # open short position
            if self._open_position(position='sell', iterator=v):
                self.positionData = v.to_dict()
                self.status = 'open'
            if self.status == 'open':
                self.countLag += 1

    def do_backtest(self, exitPosition, lag, comission, reverse=False):
        self.df.loc[:, 'return'] = 0
        self.df.loc[:, 'cumsum'] = 0
        self.df.loc[:, 'chg'] = self.df['Close'].pct_change(1)

        if exitPosition == "signal":
            self.exit_by_signal()

        if exitPosition == 'take':
            self.exit_by_take()

        if exitPosition == 'lag':
            self.exit_by_lag(lag)

        # Reverse
        if reverse:
            self.df.loc[:, 'return'] = self.df['return'] / -1

        # Commisions
        self.df.loc[:, 'return'] = np.where(
            self.df['return'] != 0, self.df['return'] - comission,
            self.df['return'])

        # Cumsum result
        self.df.loc[:, 'cumsum'] = self.df['return'].cumsum()
        return self.df


class Backtest:
    def __init__(self):
        self.position_data = {}
        self.status = 'close'
        self.count_lag = 0

    def exit_by_signal(self, data, take_profit, stop_loss, comission):
        data = data.copy()
        data.loc[:, 'return'] = 0
        data.loc[:, 'comission'] = 0
        data.loc[:, 'status'] = 'no'
        data.loc[:, 'holding'] = 0

        for i, v in data.iterrows():
            if self.status == 'open':
                profit = (v['close'] / self.position_data['close']) - 1
                self.count_lag += 1
                # Money managment
                # close long
                if (self.position_data['signal'] == 1) &\
                    ((profit >= take_profit) |
                        (profit < stop_loss)) &\
                        (self.status == 'open'):
                    data.loc[i, 'return'] = profit - comission
                    data.loc[i, 'comission'] = comission * \
                        self.position_data['close']
                    data.loc[i, 'status'] = 'mm'
                    data.loc[i, 'holding'] = self.count_lag
                    self.position_data = {}
                    self.status = 'close'
                    self.count_lag = 0
                    continue
                # close short
                if (self.position_data['signal'] == -1) &\
                    ((profit / -1 >= take_profit) |
                     (profit / -1 < stop_loss)) &\
                        (self.status == 'open'):
                    data.loc[i, 'return'] = profit / -1 - comission
                    data.loc[i, 'comission'] = comission * \
                        self.position_data['close']
                    data.loc[i, 'status'] = 'mm'
                    data.loc[i, 'holding'] = self.count_lag
                    self.position_data = {}
                    self.status = 'close'
                    self.count_lag = 0
                    continue
                # Close main position
                # close long
                if (self.position_data['signal'] == 1) &\
                        (v['signal'] == -1) & (self.status == 'open'):
                    data.loc[i, 'return'] = profit - comission
                    data.loc[i, 'comission'] = comission * \
                        self.position_data['close']
                    data.loc[i, 'status'] = 'close long'
                    data.loc[i, 'holding'] = self.count_lag
                    self.position_data = {}
                    self.status = 'close'
                    self.count_lag = 0
                    continue
                # close short
                if (self.position_data['signal'] == -1) &\
                        (v['signal'] == 1) & (self.status == 'open'):
                    data.loc[i, 'return'] = profit / -1 - comission
                    data.loc[i, 'comission'] = comission * \
                        self.position_data['close']
                    data.loc[i, 'status'] = 'close short'
                    data.loc[i, 'holding'] = self.count_lag
                    self.position_data = {}
                    self.status = 'close'
                    self.count_lag = 0
                    continue
            # Open position
            if self.status == 'close':
                # open long
                if (v['signal'] == 1) & (self.status == 'close'):
                    data.loc[i, 'status'] = 'open long'
                    self.position_data = dict(v)
                    self.status = 'open'
                    continue
                # open short
                if (v['signal'] == -1) & (self.status == 'close'):
                    data.loc[i, 'status'] = 'open short'
                    self.position_data = dict(v)
                    self.status = 'open'
                    continue
        data.loc[:, 'cumsum'] = data['return'].cumsum()
        data.loc[:, 'total_comission'] = data['comission'].cumsum()
        return data

    def exit_by_take(self, data, take_profit, stop_loss, comission):
        data = data.copy()
        data.loc[:, 'return'] = 0
        data.loc[:, 'comission'] = 0
        data.loc[:, 'status'] = 'no'
        data.loc[:, 'holding'] = 0

        for i, v in data.iterrows():
            if self.status == 'open':
                self.count_lag += 1
                profit = (v['close'] / self.position_data['close']) - 1
                # Money managment
                # close long
                if (self.position_data['signal'] == 1) &\
                    ((profit >= take_profit) |
                        (profit < stop_loss)) &\
                        (self.status == 'open'):
                    data.loc[i, 'return'] = profit - comission
                    data.loc[i, 'comission'] = comission * \
                        self.position_data['close']
                    data.loc[i, 'status'] = 'mm'
                    data.loc[i, 'holding'] = self.count_lag
                    self.position_data = {}
                    self.status = 'close'
                    self.count_lag = 0
                    continue
                # close short
                if (self.position_data['signal'] == -1) &\
                    ((profit / -1 >= take_profit) |
                     (profit / -1 < stop_loss)) &\
                        (self.status == 'open'):
                    data.loc[i, 'return'] = profit / -1 - comission
                    data.loc[i, 'comission'] = comission * \
                        self.position_data['close']
                    data.loc[i, 'status'] = 'mm'
                    data.loc[i, 'holding'] = self.count_lag
                    self.position_data = {}
                    self.status = 'close'
                    self.count_lag = 0
                    continue
            # Open position
            if self.status == 'close':
                # open long
                if (v['signal'] == 1) & (self.status == 'close'):
                    data.loc[i, 'status'] = 'open long'
                    self.position_data = dict(v)
                    self.status = 'open'
                    continue
                # open short
                if (v['signal'] == -1) & (self.status == 'close'):
                    data.loc[i, 'status'] = 'open short'
                    self.position_data = dict(v)
                    self.status = 'open'
                    continue
        data.loc[:, 'cumsum'] = data['return'].cumsum()
        data.loc[:, 'total_comission'] = data['comission'].cumsum()
        return data

    def exit_by_lag(self, data, take_profit, stop_loss, lag, comission):
        data = data.copy()
        data.loc[:, 'return'] = 0
        data.loc[:, 'comission'] = 0
        data.loc[:, 'status'] = 'no'
        data.loc[:, 'holding'] = 0
        for i, v in data.iterrows():
            if self.status == 'open':
                self.count_lag += 1
                profit = (v['close'] / self.position_data['close']) - 1
                # Money managment
                # close long
                if (self.position_data['signal'] == 1) &\
                    ((profit >= take_profit) |
                        (profit < stop_loss)) &\
                        (self.status == 'open'):
                    data.loc[i, 'return'] = profit - comission
                    data.loc[i, 'comission'] = comission * \
                        self.position_data['close']
                    data.loc[i, 'status'] = 'mm'
                    data.loc[i, 'holding'] = self.count_lag
                    self.position_data = {}
                    self.status = 'close'
                    self.count_lag = 0
                    continue
                # close short
                if (self.position_data['signal'] == -1) &\
                    ((profit / -1 >= take_profit) |
                     (profit / -1 < stop_loss)) &\
                        (self.status == 'open'):
                    data.loc[i, 'return'] = profit / -1 - comission
                    data.loc[i, 'comission'] = comission * \
                        self.position_data['close']
                    data.loc[i, 'status'] = 'mm'
                    data.loc[i, 'holding'] = self.count_lag
                    self.position_data = {}
                    self.status = 'close'
                    self.count_lag = 0
                    continue
                # Close main position
                # close long
                if (self.position_data['signal'] == 1) &\
                        (self.count_lag == lag) & (self.status == 'open'):
                    data.loc[i, 'return'] = profit - comission
                    data.loc[i, 'comission'] = comission * \
                        self.position_data['close']
                    data.loc[i, 'status'] = 'close long'
                    data.loc[i, 'holding'] = self.count_lag
                    self.position_data = {}
                    self.status = 'close'
                    self.count_lag = 0
                    continue
                # close short
                if (self.position_data['signal'] == -1) &\
                        (self.count_lag == lag) & (self.status == 'open'):
                    data.loc[i, 'return'] = profit / -1 - comission
                    data.loc[i, 'comission'] = comission * \
                        self.position_data['close']
                    data.loc[i, 'status'] = 'close short'
                    data.loc[i, 'holding'] = self.count_lag
                    self.position_data = {}
                    self.status = 'close'
                    self.count_lag = 0
                    continue

            # Open position
            if self.status == 'close':
                # open long
                if (v['signal'] == 1) & (self.status == 'close') &\
                        (self.count_lag == 0):
                    data.loc[i, 'status'] = 'open long'
                    self.position_data = dict(v)
                    self.status = 'open'
                    continue
                # open short
                if (v['signal'] == -1) & (self.status == 'close') &\
                        (self.count_lag == 0):
                    data.loc[i, 'status'] = 'open short'
                    self.position_data = dict(v)
                    self.status = 'open'
                    continue
        data.loc[:, 'cumsum'] = data['return'].cumsum()
        data.loc[:, 'total_comission'] = data['comission'].cumsum()
        return data

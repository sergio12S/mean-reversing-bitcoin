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

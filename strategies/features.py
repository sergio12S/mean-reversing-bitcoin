import numpy as np


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

        return temp['signal'].values

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

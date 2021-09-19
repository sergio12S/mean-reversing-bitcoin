from redistimeseries.client import Client
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import pandas as pd
import numpy as np

rts = Client(host='135.181.30.253', port=6379)
timeframe = 60000
# KEY = 'TWITTER:SENTIMENT'
KEY = 'LONGRATE:BTCUSDT'
twitter = dict(rts.range(KEY, from_time=0,
                         to_time=-1,  aggregation_type='sum',
                         bucket_size_msec=timeframe))
twitter = [{'timestamp': i, 'values': v} for i, v in twitter.items()]
twitter = pd.DataFrame(twitter)
twitter['timestamp'] = twitter['timestamp'] / 1000
twitter = twitter.set_index('timestamp')
twitter.columns = ['twitter']
# twitter['timestamp'] = pd.to_datetime(twitter['timestamp'])

btc = dict(rts.range('INTRADAYPRICES:BTCUSDT', from_time=0,
                     to_time=-1,  aggregation_type='sum',
                     bucket_size_msec=timeframe))
btc = [{'timestamp': i, 'values': v} for i, v in btc.items()]
btc = pd.DataFrame(btc)
btc['timestamp'] = btc['timestamp'] / 1000
btc = btc.set_index('timestamp')
btc.columns = ['btc']
# btc['timestamp'] = pd.to_datetime(btc['timestamp'])

data = pd.concat([btc, twitter], axis=1)
# data.plot(secondary_y=['twitter'])
data['y'] = data['btc'].shift(-1).pct_change()
data['y'] = np.where(data['y'] > 0, 1, 0)
# data[['y', 'twitter']].plot.scatter(x='twitter', y='y')
data = data.dropna()

x_train, x_test, y_train, y_test = train_test_split(
    data['twitter'].values, data['y'].values, test_size=0.25, random_state=0)

logisticRegr = LogisticRegression()
logisticRegr.fit(x_train.reshape(-1, 1), y_train)
predict = logisticRegr.predict(x_test.reshape(-1, 1))
metrics.confusion_matrix(y_test, predict)
np.sum(predict == y_test) / len(y_test)

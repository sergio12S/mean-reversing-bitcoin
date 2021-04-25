import pickle
from dataset import Dataset
from strategy import MlDt
import datetime
import time
import numpy as np


while True:
    print("Update data:", datetime.now())
    sleep = 5 - datetime.now().minute % 5
    if sleep == 5:
        TICKER = "SFPUSDT"
        data_binance = Dataset()
        data = data_binance.get_data(days=90, ticker='SFPUSDT', ts='1H')
        strategy_ml = MlDt(threshold=0.003)
        strategy_ml.run_strategy(data=data)
        time.sleep(sleep * 60)
    else:
        time.sleep(sleep * 60)

# Dubugging
features = strategy_ml._create_features(data)
loaded_model = pickle.load(
    open('ml_model_tree', 'rb'))
best_rules = strategy_ml._best_rules()
features = strategy_ml._optimal_parameter(data=features,
                                          sigma=best_rules['sigma'],
                                          lag=best_rules['lag'],
                                          window_ma=best_rules['window_ma'],
                                          window_std=best_rules['window_std'],
                                          plot=False)
features = features[strategy_ml.columns].dropna()
features.loc[:, 'predict'] = loaded_model.predict(
    features.loc[:, strategy_ml.columns].values)

features.loc[:, 'signal'] = np.where(
    ((features['predict'] > strategy_ml.threshold) & (features['signal'] == 1)) |
    ((features['predict'] < -strategy_ml.threshold)
     & (features['signal'] == -1)),
    features['signal'], 0)
features.tail(1)

signals = strategy_ml.do_signals(data)

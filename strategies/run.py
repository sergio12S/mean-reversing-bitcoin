from dataset import Dataset
from strategy import MlDt
from datetime import datetime
import time


while True:
    print("Update data:", datetime.now())
    sleep = 5 - datetime.now().minute % 5
    if sleep == 5:
        TICKER = "BTCUSDT"
        data_binance = Dataset()
        data = data_binance.get_data(days=90, ticker=TICKER, ts='1H')
        # ? Check settings in settings.ipynb to manage it
        strategy_ml = MlDt(threshold=0.003)
        # Check settings in __ini__ from startegy.py (money_managment)
        strategy_ml.run_strategy(data=data)
        time.sleep(sleep * 60)
    else:
        time.sleep(sleep * 60)

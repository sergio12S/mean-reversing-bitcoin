from dotenv import dotenv_values
import pandas as pd
from binance.client import Client
import datetime
config = dotenv_values(".env")
client = Client(config.get('KEY'), config.get('SECKET_KEY'))


class Dataset:
    def get_data(self, days, ticker, ts="1H"):
        '''
        https://python-binance.readthedocs.io/en/latest/binance.html
        '''
        end = datetime.datetime.now()

        end = end - datetime.timedelta(days=0)
        start = end - datetime.timedelta(days=days)
        end = end.strftime('%d %b, %Y')
        start = start.strftime('%d %b, %Y')
        if ts not in ['1H', '30m', '5m']:
            print('Imput ts: 1H or 5m or 30m')

        if ts == "1H":
            klines = client.get_historical_klines(
                ticker, Client.KLINE_INTERVAL_1HOUR, start, end)
        if ts == "5m":
            klines = client.get_historical_klines(
                ticker, Client.KLINE_INTERVAL_5MINUTE, start, end)
        if ts == "30m":
            klines = client.get_historical_klines(
                ticker, Client.KLINE_INTERVAL_30MINUTE, start, end)

        data = pd.DataFrame(data=[row[1:7] for row in klines],
                            columns=[
            "open", "high", "low", "close", "volume", "time"]
        ).set_index("time")
        data.index = pd.to_datetime(data.index + 1, unit='ms')
        data = data.sort_index()
        data = data.apply(pd.to_numeric, axis=1)
        return data

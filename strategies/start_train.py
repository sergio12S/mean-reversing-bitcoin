"""
Module to use Agent for different tickers.

to use run the command: python start_train.py run-ticker N
wher N is cryptocurency ticker
"""
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from reinforce_learning import Agent
from dataset import Dataset
import tensorflow as tf
from features import StrategiesSignals
import click


gpu_num = 0
if tf.test.gpu_device_name():
    tf.config.experimental.set_memory_growth(
        tf.config.list_physical_devices('GPU')[gpu_num], True)


@click.group()
@click.pass_context
def cli(ctx):
    """Click group."""
    pass


@click.command()
@click.argument('ticker')
def run_ticker(ticker):
    """Start Ahent for given ticker."""
    ('Ticker set to {}'.format(ticker))
    TICKER = str(ticker)

    # Dowmload data
    data_binance = Dataset()
    data = data_binance.get_data(days=90, ticker=TICKER, ts='1H')

    # Create features
    data['mean_reversing_sigma'] = StrategiesSignals.mean_reversing_sigma(
        data=data,
        sigma=1,
        lag=1,
        window_ma=100,
        window_std=100
    )
    data['mean_reversin'] = StrategiesSignals.mean_reversing(data=data,
                                                             window=200)
    data['momentum'] = StrategiesSignals.momentum(df=data['close'].values,
                                                  seq=2)

    data['reward'] = data['close'].pct_change().shift(-1)

    scaler = MinMaxScaler()
    scaler.fit(data.drop(['reward'], axis=1).values)
    train = pd.DataFrame(
        scaler.transform(data
                         .drop(['reward'], axis=1)
                         .values)
    )
    train.loc[:, 'time'] = data.index
    train.loc[:, 'open_price'] = data['close'].values
    train.loc[:, 'close_price'] = data['close'].shift(-1).values
    train.loc[:, 'reward'] = train['close_price'] - train['open_price']

    BATCH_SIZE = 200
    MEMORY_SIZE = 2000  # Experiece
    ACTION_SPACE = [0, 1, 2]  # short, hold, buy
    EPISODES = 20  # Hiw many times we lear RF
    X_VAR = train.drop(
        ['reward', 'open_price', 'close_price', 'time'], axis=1).columns
    WINDOW = len(X_VAR)
    Y_VAR = ['reward']
    ARCITECTURE = (256, 256)

    trader = Agent(
        name='{}_hour_bitcoin_test'.format(TICKER),
        window=0,
        type_model="trader_1",  # forward_net, trader_0, trader_1
        data=train.dropna(),
        window_size=WINDOW,
        batch_size=BATCH_SIZE,
        action_space=ACTION_SPACE,
        episodes=EPISODES,
        memory_size=MEMORY_SIZE,
        X_var=X_VAR,
        Y_var=Y_VAR,
        architecture=ARCITECTURE
    )
    trader.train(load_model=False)


cli.add_command(run_ticker)
if __name__ == '__main__':
    cli(obj={})

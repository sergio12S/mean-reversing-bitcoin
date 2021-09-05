from redistimeseries.client import Client
from binance import ThreadedWebsocketManager


rts = Client(host='127.0.0.1', port=6379)


def handle_socket_message(msg):
    if not msg:
        return
    rts.add('INTRADAYPRICES:BTCUSDT', msg.get(
        'k').get('t'), msg.get('k').get('c'))


def main():
    twm = ThreadedWebsocketManager()
    twm.start()
    twm.start_kline_socket(
        callback=handle_socket_message, symbol='BTCUSDT')


if __name__ == '__main__':
    main()

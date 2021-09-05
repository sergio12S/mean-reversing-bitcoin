from redistimeseries.client import Client


rts = Client(host='127.0.0.1', port=6379)

rts.create(
    'INTRADAYPRICES:BTCUSDT',
    labels={
        'SYMBOL': 'BTCUSDT',
        'DESC': 'SHARE_PRICE',
        'PRICETYPE': 'INTRADAY',
        'COIN': 'BITCOIN'
    },
    duplicate_policy='last'
)

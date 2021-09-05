from redistimeseries.client import Client
from datetime import datetime, timedelta


# server = '135.181.30.253'
server = '127.0.0.1'
rts = Client(host=server, port=6379)
rts.info('INTRADAYPRICES:BTCUSDT').__dict__


# Get last data
rts.get('INTRADAYPRICES:BTCUSDT')
# Get all data
print(rts.range('INTRADAYPRICES:BTCUSDT', 0, -1))
# Get data between timestamp from start to end
print(rts.range('INTRADAYPRICES:BTCUSDT', '1630844040000', '1630844220000'))
rts.range('INTRADAYPRICES:BTCUSDT', 0, -1,
          aggregation_type='avg', bucket_size_msec=5)

end_t = str(int(datetime.timestamp(datetime.now())))
start_t = str(int(datetime.timestamp(datetime.now() - timedelta(hours=6))))
print(rts.range('INTRADAYPRICES:BTCUSDT', start_t + '000', end_t + '000'))

from loguru import logger
from db import current_session, Strategies
from datetime import timedelta
from binance.client import Client
KEY = ''
SECRET_KEY = ''
client = Client(KEY, SECRET_KEY)


class Manager:
    def __init__(self, name_strategy, data, take_profit, stop_loss,
                 lag,
                 ticker, size):
        self.name_strategy = name_strategy
        self.data = data
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        self.lag = lag
        self.ticker = ticker
        self.size = size

    def _percentChange(self, openPosition, closePosition):
        return (float(closePosition) - openPosition) / abs(openPosition)

    def check_status(self):
        events = current_session.query(Strategies).with_entities(
            Strategies.Strategy,
            Strategies.Status,
            Strategies.Time,
            Strategies.Open,
            Strategies.Close,
            Strategies.Lag,
            Strategies.Signal,
            Strategies.Rule,
            Strategies.Result,
            Strategies.Cumsum
        ).filter(
            Strategies.Strategy == self.name_strategy
        ).order_by(Strategies.Time.desc()).first()
        return events

    def open_long(self):
        new_trade = Strategies(
            Strategy=self.name_strategy,
            Status="open",
            Time=self.data.index.to_pydatetime()[0],
            Open=float(self.data['close']),
            Lag=int(self.lag),
            Signal=int(self.data['signal'])
        )
        current_session.add(new_trade)
        current_session.commit()
        logger.info(f'Open long in {self.name_strategy}')
        status = {
            'name': self.name_strategy,
            'status': 'open',
            'time': self.data.index.to_pydatetime()[0],
            'open': float(self.data['close']),
            'signal': int(self.data['signal'])
        }
        try:
            check_orders = client.get_open_orders(symbol=self.ticker)
            if len(check_orders) == 1:
                cancel = client.cancel_order(
                    symbol=self.ticker, orderId=check_orders[0].get('orderId'))
                if cancel.get('status') == 'CANCELED':
                    print('Canceled previous order')
            client.order_market_buy(symbol=self.ticker, quantity=self.size)
        except Exception as e:
            print(e)

        return status

    def open_short(self):
        new_trade = Strategies(
            Strategy=self.name_strategy,
            Status="open",
            Time=self.data.index.to_pydatetime()[0],
            Open=float(self.data['close']),
            Lag=int(self.lag),
            Signal=int(self.data['signal'])
        )
        current_session.add(new_trade)
        current_session.commit()
        logger.info(f'Open short in {self.name_strategy}')
        status = {
            'name': self.name_strategy,
            'status': 'open',
            'time': self.data.index.to_pydatetime()[0],
            'open': float(self.data['close']),
            'signal': int(self.data['signal'])
        }
        try:
            check_orders = client.get_open_orders(symbol=self.ticker)
            if len(check_orders) == 1:
                cancel = client.cancel_order(
                    symbol=self.ticker, orderId=check_orders[0].get('orderId'))
                if cancel.get('status') == 'CANCELED':
                    print('Canceled previous order')
            client.order_market_sell(symbol=self.ticker, quantity=self.size)
        except Exception as e:
            print(e)
        return status

    def close_long(self, profit):
        current_session.query(Strategies).filter(
            Strategies.Status == "open",
            Strategies.Strategy == self.name_strategy
        ).update(
            {
                "Close": float(self.data['close']),
                "Status": "close",
                "Result": profit
            }, synchronize_session=False
        )
        current_session.commit()
        logger.info(f'Close long in {self.name_strategy}, profit: {profit}')
        status = {
            'name': self.name_strategy,
            'status': 'close',
            'time': self.data.index.to_pydatetime()[0],
            'close': float(self.data['close']),
            'signal': int(self.data['signal']),
            'profit': profit
        }
        return status

    def close_short(self, profit):
        current_session.query(Strategies).filter(
            Strategies.Status == "open",
            Strategies.Strategy == self.name_strategy
        ).update(
            {
                "Close": float(self.data['close']),
                "Status": "close",
                "Result": profit / -1
            }, synchronize_session=False
        )
        current_session.commit()
        logger.info(f'Close short in {self.name_strategy}, profit: {profit}')
        status = {
            'name': self.name_strategy,
            'status': 'close',
            'time': self.data.index.to_pydatetime()[0],
            'close': float(self.data['close']),
            'signal': int(self.data['signal']),
            'profit': profit
        }
        return status

    def exit_position(self, profit, event):
        if event.Signal == 1:
            self.close_long(profit=profit)
        if event.Signal == -1:
            self.close_short(profit=profit)

    def send_to_telegram(self):
        pass

    def manage(self):
        event = self.check_status()
        if not event:
            # open long
            if int(self.data['signal'] == 1):
                status = self.open_long()
                return status
            # open short
            if int(self.data['signal'] == -1):
                status = self.open_short()
                return status
        if event:
            if event.Status == 'close':
                # open long
                if int(self.data['signal'] == 1):
                    status = self.open_long()
                    return status
                # close long
                if int(self.data['signal'] == -1):
                    status = self.open_short()
                    return status
            if event.Status == 'open':
                current_profit = self._percentChange(
                    openPosition=event.Open,
                    closePosition=self.data['close']
                )
                time_delta = event.Time + timedelta(minutes=self.lag * 5)
                # close long by lag Rules # 1
                if self.data.index > time_delta:
                    status = self.exit_position(
                        profit=current_profit, event=event)
                    return status
                # close by take profit # Rule s
                if current_profit >= self.take_profit:
                    status = self.exit_position(
                        profit=current_profit, event=event)
                    return status
                # close by stop loss
                if current_profit <= self.stop_loss:
                    status = self.exit_position(
                        profit=current_profit, event=event)
                    return status
                # close by signal
                if int(self.data['signal'] != event.Signal):
                    if int(self.data['signal']) == 1 or \
                            int(self.data['signal']) == -1:
                        status = self.exit_position(
                            profit=current_profit, event=event)
                        return status

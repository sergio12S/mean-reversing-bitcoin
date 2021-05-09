# Plan to make quant strategy

1. [x] Adapt the strategy to different instruments at different time periods.
1.1 [x] Getting data from the official API binance.
2. [x] Make methods for testing statistical hypotheses.
3. Make a method to test the hypothesis that the data has predictive power.
4. Automatic updating of rules for a trading strategy when market conditions change. If trading becomes random based on statistical hypotheses, then it is necessary to rebalance new patterns.
5. Determination of the optimal memory period for the market.
6. Make the mode of enabling and disabling machine learning for a trading strategy.
7. Create class to coleect features with predictive power values (mean reversing, seq, candle, ta, volume)

## We make big research

1. [x] Create mean reversing signal
2. [x] Create momentum signal
3. Create candle patterns signal
4. Create TA signal. Create ATR
5. Create volume signal


## Runing strategy

1. [x] Run setings.ipynb for create rules and ML models
2. [x] Run strategy run.py
3. [x] Create database for strategies
4. Automatic retrain data if negative profit ror the last 2 days
5. Optimal money managment


## Today
[x] 1. Automatic money managments

## Plan to improve strategy
1. [x] Use price data
2. Use volume data
3. Use twitter data
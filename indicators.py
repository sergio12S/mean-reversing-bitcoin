import pandas as pd
import numpy as np


def rate_of_change(df, n):
    M = df['close'].diff(n - 1)
    N = df['close'].shift(n - 1)
    ROC = pd.Series(M / N, name='ROC_' + str(n))
    return df.join(ROC), ['ROC_' + str(n)]


def moving_average(df, n):
    MA = pd.Series(df['close'].rolling(
        n, min_periods=n).mean(), name='MA_' + str(n))
    return df.join(MA), ['MA_' + str(n)]


def average_true_range(df, n):
    i = 0
    TR_l = [0]
    while i < df.index[-1]:
        TR = max(df.loc[i + 1, 'high'], df.loc[i, 'close']) - \
            min(df.loc[i + 1, 'low'], df.loc[i, 'close'])
        TR_l.append(TR)
        i = i + 1
    TR_s = pd.Series(TR_l)
    ATR = pd.Series(TR_s.ewm(span=n, min_periods=n).mean(),
                    name='ATR_' + str(n))
    return df.join(ATR), ['ATR_' + str(n)]


def bollinger_bands(df, n):
    MA = pd.Series(df['close'].rolling(n, min_periods=n).mean())
    MSD = pd.Series(df['close'].rolling(n, min_periods=n).std())
    b1 = 4 * MSD / MA
    B1 = pd.Series(b1, name='BollingerB_' + str(n))
    df = df.join(B1)
    b2 = (df['close'] - MA + 2 * MSD) / (4 * MSD)
    B2 = pd.Series(b2, name='Bollinger%b_' + str(n))
    return df.join(B2), ['BollingerB_' + str(n), 'Bollinger%b_' + str(n)]


def stochastic_oscillator_d(df, n):
    SOk = pd.Series((df['close'] - df['low']) /
                    (df['high'] - df['low']), name='SO%k')
    SOd = pd.Series(SOk.ewm(span=n, min_periods=n).mean(),
                    name='SO%d_' + str(n))
    return df.join(SOd), ['SO%d_' + str(n)]


def average_directional_movement_index(df, n):
    i = 0
    UpI = []
    DoI = []
    while i + 1 <= df.index[-1]:
        UpMove = df.loc[i + 1, 'high'] - df.loc[i, 'high']
        DoMove = df.loc[i, 'low'] - df.loc[i + 1, 'low']
        if UpMove > DoMove and UpMove > 0:
            UpD = UpMove
        else:
            UpD = 0
        UpI.append(UpD)
        if DoMove > UpMove and DoMove > 0:
            DoD = DoMove
        else:
            DoD = 0
        DoI.append(DoD)
        i = i + 1
    i = 0
    TR_l = [0]
    while i < df.index[-1]:
        TR = max(df.loc[i + 1, 'high'], df.loc[i, 'close']) - \
            min(df.loc[i + 1, 'low'], df.loc[i, 'close'])
        TR_l.append(TR)
        i = i + 1
    TR_s = pd.Series(TR_l)
    ATR = pd.Series(TR_s.ewm(span=n, min_periods=n).mean())
    UpI = pd.Series(UpI)
    DoI = pd.Series(DoI)
    PosDI = pd.Series(UpI.ewm(span=n, min_periods=n).mean() /
                      ATR, name="PosDI_" + str(n))
    NegDI = pd.Series(DoI.ewm(span=n, min_periods=n).mean() /
                      ATR, name="NegDI_" + str(n))
    ADX = pd.Series((abs(PosDI - NegDI) / (PosDI + NegDI)
                     ).ewm(span=n, min_periods=n).mean(), name='ADX_' + str(n))
    return df.\
        join(PosDI)\
        .join(NegDI).join(ADX), [
            "PosDI_" + str(n), "NegDI_" + str(n), 'ADX_' + str(n)]


def kst_oscillator(df, r1=10, r2=15, r3=20, r4=30, n1=10, n2=10, n3=10, n4=15):
    M = df['close'].diff(r1 - 1)
    N = df['close'].shift(r1 - 1)
    ROC1 = M / N
    M = df['close'].diff(r2 - 1)
    N = df['close'].shift(r2 - 1)
    ROC2 = M / N
    M = df['close'].diff(r3 - 1)
    N = df['close'].shift(r3 - 1)
    ROC3 = M / N
    M = df['close'].diff(r4 - 1)
    N = df['close'].shift(r4 - 1)
    ROC4 = M / N
    col_name = 'KST_'+str(r1)+'_'+str(r2)+'_'+str(r3)+'_' + \
        str(r4)+'_'+str(n1)+'_'+str(n2)+'_'+str(n3)+'_'+str(n4)
    KST = pd.Series(
        ROC1.rolling(n1).sum() + ROC2.rolling(n2).sum() * 2 +
        ROC3.rolling(n3).sum() * 3 + ROC4.rolling(n4).sum() * 4,
        name=col_name)
    return df.join(KST), [col_name]


def standard_deviation(df, n):
    std = pd.Series(df['close'].rolling(
        n, min_periods=n).std(), name='STD_' + str(n))
    return df.join(std), ['STD_' + str(n)]


def EVM(df):
    dm = 0.5 * (df['high'] + df['low'] -
                df['high'].shift(1) + df['low'].shift(1))
    br = df['volumeto'] / 100000000 / (df['high'] - df['low'])
    EVM = pd.Series(dm / br, name="EVM")
    return df.join(EVM), ["EVM"]


def trix(df, n):
    EX1 = df['close'].ewm(span=n, min_periods=n).mean()
    EX2 = EX1.ewm(span=n, min_periods=n).mean()
    EX3 = EX2.ewm(span=n, min_periods=n).mean()
    i = 0
    ROC_l = [np.nan]
    while i + 1 <= df.index[-1]:
        ROC = (EX3[i + 1] - EX3[i]) / EX3[i]
        ROC_l.append(ROC)
        i = i + 1
    Trix = pd.Series(ROC_l, name='Trix_' + str(n))
    return df.join(Trix), ['Trix_' + str(n)]


def macd(df, n_fast, n_slow):
    EMAfast = pd.Series(df['close'].ewm(
        span=n_fast, min_periods=n_slow).mean())
    EMAslow = pd.Series(df['close'].ewm(
        span=n_slow, min_periods=n_slow).mean())
    MACD = pd.Series(EMAfast - EMAslow, name='MACD_' +
                     str(n_fast) + '_' + str(n_slow))
    MACDsign = pd.Series(MACD.ewm(span=9, min_periods=9).mean(
    ), name='MACDsign_' + str(n_fast) + '_' + str(n_slow))
    MACDdiff = pd.Series(MACD - MACDsign, name='MACDdiff_' +
                         str(n_fast) + '_' + str(n_slow))
    return df.join(MACD).join(MACDsign).join(MACDdiff), \
        ['MACD_'+str(n_fast)+'_'+str(n_slow), 'MACDsign_'+str(n_fast) +
         '_'+str(n_slow), 'MACDdiff_'+str(n_fast)+'_'+str(n_slow)]


def mass_index(df):
    Range = df['high'] - df['low']
    EX1 = Range.ewm(span=9, min_periods=9).mean()
    EX2 = EX1.ewm(span=9, min_periods=9).mean()
    Mass = EX1 / EX2
    MassI = pd.Series(Mass.rolling(25).sum(), name='Mass_Index')
    return df.join(MassI), ['Mass_Index']


def vortex_indicator(df, n):
    i = 0
    TR = [0]
    while i < df.index[-1]:
        Range = max(df.loc[i + 1, 'high'], df.loc[i, 'close']) - \
            min(df.loc[i + 1, 'low'], df.loc[i, 'close'])
        TR.append(Range)
        i = i + 1
    i = 0
    VM = [0]
    while i < df.index[-1]:
        Range = abs(df.loc[i + 1, 'high'] - df.loc[i, 'low']) - \
            abs(df.loc[i + 1, 'low'] - df.loc[i, 'high'])
        VM.append(Range)
        i = i + 1
    VI = pd.Series(pd.Series(VM).rolling(n).sum() /
                   pd.Series(TR).rolling(n).sum(), name='Vortex_' + str(n))
    return df.join(VI), ['Vortex_' + str(n)]


def relative_strength_index(df, n):
    i = 0
    UpI = [0]
    DoI = [0]
    while i + 1 <= df.index[-1]:
        UpMove = df.loc[i + 1, 'high'] - df.loc[i, 'high']
        DoMove = df.loc[i, 'low'] - df.loc[i + 1, 'low']
        if UpMove > DoMove and UpMove > 0:
            UpD = UpMove
        else:
            UpD = 0
        UpI.append(UpD)
        if DoMove > UpMove and DoMove > 0:
            DoD = DoMove
        else:
            DoD = 0
        DoI.append(DoD)
        i = i + 1
    UpI = pd.Series(UpI)
    DoI = pd.Series(DoI)
    PosDI = pd.Series(UpI.ewm(span=n, min_periods=n).mean())
    NegDI = pd.Series(DoI.ewm(span=n, min_periods=n).mean())
    RSI = pd.Series(PosDI / (PosDI + NegDI), name='RSI_' + str(n))
    return df.join(RSI), ['RSI_' + str(n)]


def true_strength_index(df, r, s):
    M = pd.Series(df['close'].diff(1))
    aM = abs(M)
    EMA1 = pd.Series(M.ewm(span=r, min_periods=r).mean())
    aEMA1 = pd.Series(aM.ewm(span=r, min_periods=r).mean())
    EMA2 = pd.Series(EMA1.ewm(span=s, min_periods=s).mean())
    aEMA2 = pd.Series(aEMA1.ewm(span=s, min_periods=s).mean())
    TSI = pd.Series(EMA2 / aEMA2, name='TSI_' + str(r) + '_' + str(s))
    return df.join(TSI), ['TSI_' + str(r) + '_' + str(s)]


def accumulation_distribution(df, n):
    ad = (2 * df['close'] - df['high'] - df['low']) / \
        (df['high'] - df['low']) * df['volumeto']
    M = ad.diff(n - 1)
    N = ad.shift(n - 1)
    ROC = M / N
    AD = pd.Series(ROC, name='Acc/Dist_ROC_' + str(n))
    return df.join(AD), ['Acc/Dist_ROC_' + str(n)]


def chaikin_oscillator(df):
    ad = (2 * df['close'] - df['high'] - df['low']) / \
        (df['high'] - df['low']) * df['volumeto']
    Chaikin = pd.Series(ad.ewm(span=3, min_periods=3).mean(
    ) - ad.ewm(span=10, min_periods=10).mean(), name='Chaikin')
    return df.join(Chaikin), ['Chaikin']


def money_flow_index(df, n):
    PP = (df['high'] + df['low'] + df['close']) / 3
    i = 0
    PosMF = [0]
    while i < df.index[-1]:
        if PP[i + 1] > PP[i]:
            PosMF.append(PP[i + 1] * df.loc[i + 1, 'volumeto'])
        else:
            PosMF.append(0)
        i = i + 1
    PosMF = pd.Series(PosMF)
    TotMF = PP * df['volumeto']
    MFR = pd.Series(PosMF / TotMF)
    MFI = pd.Series(MFR.rolling(n, min_periods=n).mean(), name='MFI_' + str(n))
    return df.join(MFI), ['MFI_' + str(n)]


def on_balance_volume(df, n):
    i = 0
    OBV = [0]
    while i < df.index[-1]:
        if df.loc[i + 1, 'close'] - df.loc[i, 'close'] > 0:
            OBV.append(df.loc[i + 1, 'volumeto'])
        if df.loc[i + 1, 'close'] - df.loc[i, 'close'] == 0:
            OBV.append(0)
        if df.loc[i + 1, 'close'] - df.loc[i, 'close'] < 0:
            OBV.append(-df.loc[i + 1, 'volumeto'])
        i = i + 1
    OBV = pd.Series(OBV)
    OBV_ma = pd.Series(OBV.rolling(n, min_periods=n).mean(),
                       name='OBV_' + str(n))
    return df.join(OBV_ma), ['OBV_' + str(n)]


def force_index(df, n):
    F = pd.Series(df['close'].diff(
        n) * df['volumeto'].diff(n), name='Force_' + str(n))
    return df.join(F), ['Force_' + str(n)]


def ease_of_movement(df, n):
    EoM = (df['high'].diff(1) + df['low'].diff(1)) * \
        (df['high'] - df['low']) / (2 * df['volumeto'])
    Eom_ma = pd.Series(EoM.rolling(n, min_periods=n).mean(),
                       name='EoM_' + str(n))
    return df.join(Eom_ma), ['EoM_' + str(n)]


def commodity_channel_index(df, n):
    PP = (df['high'] + df['low'] + df['close']) / 3
    CCI = pd.Series((PP - PP.rolling(n, min_periods=n).mean()) /
                    PP.rolling(n, min_periods=n).std(), name='CCI_' + str(n))
    return df.join(CCI), ['CCI_' + str(n)]


def coppock_curve(df, n):
    M = df['close'].diff(int(n * 11 / 10) - 1)
    N = df['close'].shift(int(n * 11 / 10) - 1)
    ROC1 = M / N
    M = df['close'].diff(int(n * 14 / 10) - 1)
    N = df['close'].shift(int(n * 14 / 10) - 1)
    ROC2 = M / N
    Copp = pd.Series(
        (ROC1 + ROC2).ewm(span=n, min_periods=n).mean(), name='Copp_' + str(n))
    return df.join(Copp), ['Copp_' + str(n)]


def keltner_channel(df, n):
    KelChM = pd.Series(((df['high'] + df['low'] + df['close']) / 3)
                       .rolling(n, min_periods=n).mean(),
                       name='KelChM_' + str(n))
    KelChU = pd.Series(((4 * df['high'] - 2 * df['low'] + df['close']) / 3)
                       .rolling(n, min_periods=n).mean(),
                       name='KelChU_' + str(n))
    KelChD = pd.Series(((-2 * df['high'] + 4 * df['low'] + df['close']) / 3)
                       .rolling(n, min_periods=n).mean(),
                       name='KelChD_' + str(n))
    return df\
        .join(KelChM)\
        .join(KelChU)\
        .join(KelChD), \
        ['KelChM_' + str(n), 'KelChU_' + str(n), 'KelChD_' + str(n)]


def ultimate_oscillator(df):
    i = 0
    TR_l = [0]
    BP_l = [0]
    while i < df.index[-1]:
        TR = max(df.loc[i + 1, 'high'], df.loc[i, 'close']) - \
            min(df.loc[i + 1, 'low'], df.loc[i, 'close'])
        TR_l.append(TR)
        BP = df.loc[i + 1, 'close'] - \
            min(df.loc[i + 1, 'low'], df.loc[i, 'close'])
        BP_l.append(BP)
        i = i + 1
    UltO = pd.Series((4 * pd.Series(BP_l)
                      .rolling(7).sum() / pd.Series(TR_l)
                      .rolling(7).sum()) + (
        2 * pd.Series(BP_l)
        .rolling(14).sum() / pd.Series(TR_l)
        .rolling(14).sum()) + (
        pd.Series(BP_l).rolling(28).sum() / pd.Series(TR_l)
            .rolling(28).sum()),
        name='Ultimate_Osc')
    return df.join(UltO), ['Ultimate_Osc']


def donchian_channel(df, n):
    i = 0
    dc_l = []
    while i < n - 1:
        dc_l.append(0)
        i += 1
    i = 0
    while i + n - 1 < df.index[-1]:
        dc = max(df['high'].ix[i:i + n - 1]) - min(df['low'].ix[i:i + n - 1])
        dc_l.append(dc)
        i += 1
    donchian_chan = pd.Series(dc_l, name='Donchian_' + str(n))
    donchian_chan = donchian_chan.shift(n - 1)
    return df.join(donchian_chan), ['Donchian_' + str(n)]


def DRF(df):
    DRF = pd.Series((df["high"] - df["open"] + df["close"] -
                     df["low"]) / (2 * (df["high"] - df["low"])), name="DRF")
    return df.join(DRF), ["DRF"]

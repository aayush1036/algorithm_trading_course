import numpy as np
import pandas as pd

def drawdown(return_series: pd.Series):
    """
    Takes a Time Series of asset returns and computes and returns a dataframe that contains
    1) Wealth Idex
    2) Previous Peaks
    3) Percentage of drawdowns
    """
    wealth_index = 1000*(1+return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index-previous_peaks)/previous_peaks
    return pd.DataFrame(dict(Wealth = wealth_index, Peaks = previous_peaks, Drawdowns = drawdowns))

def get_ffme_returns():
    """
    Load the Farma-French Dataset for the returns of the top and bottom deciles by Market Cap
    """
    me_m = pd.read_csv('Portfolios_Formed_on_ME_monthly_EW.csv', header=0, index_col=0, na_values = -99.99)
    rets = me_m[['Lo 10', 'Hi 10']]
    rets.columns = ['SmallCap', 'LargeCap']
    rets = rets/100
    rets.index = pd.to_datetime(rets.index, format="%Y%m").to_period('M')
    return rets

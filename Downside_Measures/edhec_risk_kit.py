import numpy as np
import pandas as pd
import scipy.stats
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

def get_hfi_returns():
    """
    Load the Edhec Hedge Fund Dataset for the returns of the top and bottom deciles by Market Cap
    """
    hfi = pd.read_csv('edhec-hedgefundindices.csv', header=0, index_col=0, parse_dates=True)
    hfi = hfi/100
    hfi.index = hfi.index.to_period('M')
    return hfi

def semideviation(r):
    """
    Returns the semideviation aka negative semideviation of r 
    r must be a series or a dataframe
    """
    is_negative = r<0  #produced the mask 
    return r[is_negative].std(ddof=0)

"""
semideviation can also be done in a more efficient way 
def semideviation(r):
    return r[r<0].std(ddof=0)
"""

def skewness(r):
    """
    Alternative method to scipy.stats.skew()
    Computes the skewness of the supplied Series or Dataframe 
    Retruns a float or a series
    """
    demeaned_r = r - r.mean()
    #use the population standard deviation so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**3).mean()
    return exp/(sigma_r**3)

def kurtosis(r):
    """
    Alternative method to scipy.stats.kurtosis()
    Computes the kurtosis of the supplied Series or Dataframe 
    Retruns a float or a series
    """
    demeaned_r = r - r.mean()
    #use the population standard deviation so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**4).mean()
    return exp/(sigma_r**4)

def is_normal(r, level = 0.01):
    """
    Applies the Jarque-Bera test to determine if a series is normal or not 
    Test is applied at the 1% level by default
    Returns true if the hypothesis of normality is accepted otherwise returns False
    """
    statistic, p_value = scipy.stats.jarque_bera(r)
    return p_value>level

def var_historic(r, level=5):
    """
    Returns the historic value at risk at a specified level 
    i.e returns the number such that the 'level' percent of the returns 
    fall below that number, and the (100-level) percent are above
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level=level) #calls this function for every column if the series is a dataframe 
    elif isinstance(r, pd.Series):
        return -np.percentile(r, level)
    else:
        raise TypeError('Expected r to be a series or a dataframe')

from scipy.stats import norm 
def var_gaussian(r, level=5, modified=False):
    """
    Returns the parametric Gaussian VaR of a series or dataframe 
    """
    #compute the z score 
    z = norm.ppf(level/100)
    if modified:
        s = skewness(r)
        k= kurtosis(r)
        z = (z+
        (z**2 - 1)*s/6 + 
        (z**3 - 3*z)*(k-3)/24 -
        (2*z**3 - 5*z)*(s**2)/36)


    return -(r.mean() + z*r.std(ddof=0))

def cvar_historic(r, level=5):
    """
    Computes the conditional VaR of series or dataframe
    """
    if isinstance(r, pd.Series):
        is_beyond = r <= -var_historic(r, level=level)
        return -r[is_beyond].mean()
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, level=level)
    else:
        raise TypeError('Expected r to be series or dataframe')
import pandas as pd
import scipy
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
import math
import statsmodels.api as sm




def drawdown(ret: pd.Series):
    """
    Takes a time series of asset returns and gives a Dataframe with columns for the wealth index, the previous
    peaks and the percentage drawdown
    """

    wealth_index = 1000*(1+ret).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks)/previous_peaks
    return pd.DataFrame(
        {
        "Wealth": wealth_index,
        "Previous Peaks": previous_peaks,
        "Drawdowns": drawdowns
        }
    )


def get_ffme_returns():
    """
    Load the Fama-French Dataset for the returns of the Top and Bottom Deciles by MarketCap.
    """

    me_m = pd.read_csv("Portfolios_Formed_on_ME_monthly_EW.csv", header=0, index_col=0, na_values=-99.99)
    rets = me_m[['Lo 10', 'Hi 10']]
    rets.columns = ['SmallCap', 'LargeCap']
    rets = rets/100
    rets.index = pd.to_datetime(rets.index, format="%Y%m").to_period('M')
    return rets


def get_fff_returns():
    """
    Load the Fama-French Research Factor Monthly Dataset
    """
    rets = pd.read_csv("data/F-F_Research_Data_Factors_m.csv",
                       header=0, index_col=0, na_values=-99.99)/100
    rets.index = pd.to_datetime(rets.index, format="%Y%m").to_period('M')
    return rets


def get_hfi_returns():
    """
    Load and format the EDHEC Hedge Fund Inde Returns
    """

    hfi = pd.read_csv("edhec-hedgefundindices.csv", header=0, index_col=0, parse_dates=True)
    hfi = hfi/100
    hfi.index = hfi.index.to_period('M')
    return hfi

#
# def get_ind_returns():
#     """
#     Load and format the Ken French 30 Industry Portfolios Value Weighted Monthly Returns.
#     """
#     ind = pd.read_csv("data\ind30_m_vw_rets.csv", header=0, index_col=0)/100
#     ind.index = pd.to_datetime(ind.index, format="%Y%m").to_period('M')
#     ind.columns = ind.columns.str.strip()
#     return ind


# def get_ind_file(filetype, ew=False):
#     """
#     Load and format the Ken French 30 Industry Portfolios files
#     """
#     known_types = ["returns", "nfirms", "size"]
#     if filetype not in known_types:
#         raise ValueError(f"filetype must be one of:{','.join(known_types)}")
#     if filetype is "returns":
#         name = "ew_rets" if ew else "vw_rets"
#         divisor = 100
#     elif filetype is "nfirms":
#         name = "nfirms"
#         divisor = 1
#     elif filetype is "size":
#         name = "size"
#         divisor = 1
#
#     ind = pd.read_csv(f"data/ind30_m_{name}.csv", header=0, index_col=0) / divisor
#     ind.index = pd.to_datetime(ind.index, format="%Y%m").to_period('M')
#     ind.columns = ind.columns.str.strip()
#     return ind



def get_ind_file(filetype, weighting="vw", n_inds=30):
    """
    Load and format the Ken French Industry Portfolios files
    Variant is a tuple of (weighting, size) where:
        weighting is one of "ew", "vw"
        number of inds is 30 or 49
    """
    if filetype == "returns":
        name = f"{weighting}_rets"
        divisor = 100
    elif filetype == "nfirms":
        name = "nfirms"
        divisor = 1
    elif filetype == "size":
        name = "size"
        divisor = 1
    else:
        raise ValueError(f"filetype must be one of: returns, nfirms, size")

    ind = pd.read_csv(f"data/ind{n_inds}_m_{name}.csv", header=0, index_col=0, na_values=-99.99) / divisor
    ind.index = pd.to_datetime(ind.index, format="%Y%m").to_period('M')
    ind.columns = ind.columns.str.strip()
    return ind



def get_ind_market_caps(n_inds=30, weights=False):
    """
    Load the industry portfolio data and derive the market caps
    """
    ind_nfirms = get_ind_nfirms(n_inds=n_inds)
    ind_size = get_ind_size(n_inds=n_inds)
    ind_mktcap = ind_nfirms * ind_size
    if weights:
        total_mktcap = ind_mktcap.sum(axis=1)
        ind_capweight = ind_mktcap.divide(total_mktcap, axis="rows")
        return ind_capweight
    #else
    return ind_mktcap



# def get_ind_returns(ew=False):
#     """
#     Load and format the Ken French 30 Industry Portfolios Value Weighted Monthly Returns
#     """
#     return get_ind_file("returns", ew=ew)

def get_ind_returns(weighting="vw", n_inds=30):
    """
    Load and format the Ken French Industry Portfolios Monthly Returns
    """
    return get_ind_file("returns", weighting=weighting, n_inds=n_inds)


# def get_ind_size():
#     """
#     """
#     ind = pd.read_csv("ind30_m_size.csv", header=0, index_col=0)
#     ind.index = pd.to_datetime(ind.index, format="%Y%m").to_period('M')
#     ind.columns = ind.columns.str.strip()
#     return ind

def get_ind_size(n_inds=30):
    """
    Load and format the Ken French 30 Industry Portfolios Average size (market cap)
    """
    return get_ind_file("size", n_inds=n_inds)



# def get_ind_nfirms():
#     """
#     """
#     ind = pd.read_csv("ind30_m_nfirms.csv", header=0, index_col=0)
#     ind.index = pd.to_datetime(ind.index, format="%Y%m").to_period('M')
#     ind.columns = ind.columns.str.strip()
#     return ind



def get_ind_nfirms(n_inds=30):
    """
    Load and format the Ken French 30 Industry Portfolios Average number of Firms
    """
    return get_ind_file("nfirms", n_inds=n_inds)




def skewness(r):
    """
    Alternative to scipy.stats.skew()
    Computes the skewness of the supplied Series or DataFrame
    Returns a float or a Series
    """

    demeaned_r = r - r.mean()
    # use the population standard deviation, so set dof=0
    # degrees of freedom=0 means do not make n-1 correction. It doesn't matter if you have
    # enough data then dividing by n or n-1 will leave reult in very slightly diff numbers
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**3).mean()
    return exp/sigma_r**3


def kurtosis(r):
    """
    Alternative to scipy.stats.kurtosis()
    Computes the Kurtosis of the supplied Series or DataFrame
    Returns a float or a Series
    """

    demeaned_r = r - r.mean()
    # use the population standard deviation, so set dof=0
    # degrees of freedom=0 means do not make n-1 correction. It doesn't matter if you have
    # enough data then dividing by n or n-1 will leave reult in very slightly diff numbers
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**4).mean()
    return exp/sigma_r**4


def compound(r):
    """
    returns the result of compounding the set of returns in r
    """
    return np.expm1(np.log1p(r).sum())


def is_normal(r, level=0.01):
    """
    Applies Jarque Bera test to determine if a Series is normal or not.
    Test is appled at the 1% level by default.
    Returns True if the hypothesis of normality is accepted. False otherwise.
    """
    statistic, p_value = scipy.stats.jarque_bera(r)
    return p_value > level


def semideviation(r):
    """
    Returns the semideviation aka negative semideviation of r.
    r must be a series or dataframe.
    """
    is_negative = r < 0
    return r[is_negative].std(ddof=0)


def var_historic(r, level=5):
    """
    Return the historic value at Risk at a specified level
    i.e; returns the number such that "level" percent of the returns fall below that number, and the (100-level) percent are above
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level=level)
    elif isinstance(r, pd.Series):
        return -np.percentile(r, level) # -ve since we always express it as a positive number
    else:
        raise TypeError("Expected r to be Series or DataFrame")


def var_gaussian(r, level=5, modified=False):
    """
    Returns the Parametric Gaussian VaR of a Series or DataFrame
    If 'modified' is True, then the modified VaR is returned using the Cornish-Fisher modification.
    """
    # Compute the z-score assuming it was Gaussian
    z = norm.ppf(level/100)
    if modified:
        # modified the Z score based on observed skewness and kurtosis
        s = skewness(r)
        k = kurtosis(r)
        z = (
                z +
                (z**2 - 1)*s/6 +
                (z**3 - 3*z)*(k-3)/24 -
                (2*z**3 - 5*z)*(s**2)/36
            )
    return -(r.mean() +z*r.std(ddof=0))


def cvar_historic(r, level=5):
    """
    Computes the Conditional VaR of Series or DataFrame
    """
    if isinstance(r, pd.Series):
        is_beyond = r <= -var_historic(r, level=level) # find returns less than historic_var.
        # - sign as var_historic is -ve
        return -r[is_beyond].mean()
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, level=level)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")


def annualize_rets(r, periods_per_year):
    """
    Annualizes a set of returns.
    We should actually infer the periods per year
    """
    compounded_growth = (1+r).prod()
    n_periods = r.shape[0]
    return compounded_growth**(periods_per_year/n_periods) - 1


def annualize_vol(r, periods_per_year):
    """
     Annualizes the volatility of a set of returns.
     We should actually infer the periods per year.
    """
    return r.std()*(periods_per_year**0.5)


def sharpe_ratio(r, riskfree_rate, periods_per_year):
    """
    Computes the annualized sharpe ratio of a set of returns
    """
    # Convert the annual risk free ratio to per period
    rf_per_period = (1 + riskfree_rate)**(1/periods_per_year) - 1
    excess_ret = r - rf_per_period
    annualized_excess_ret = annualize_rets(excess_ret, periods_per_year)
    annualized_volatility = annualize_vol(r, periods_per_year)
    return annualized_excess_ret/annualized_volatility


def portfolio_return(weights, returns):
    """
    :param weights: weights for assets
    :param returns: returns
    :return: Weights ->  Returns
    """
    return weights.T @ returns


def portfolio_vol(weights, covmat):
    """
    :param weights:
    :param covmat:
    :return: Weights -> Vol
    """
    return (np.dot(weights.T , covmat) @ weights )**0.5  #variance = weights.T @ covmat @ weights


def plot_ef2(n_points, er, cov):
    """
    Plots the 2 Asset Efficient Frontier
    :param n_points: number of points
    :param er: expected returns
    :param cov: covariance matrix
    :return: plot of efficient frontier
    """
    if er.shape[0]!=2:
        raise ValueError("plot_ef2 can only plot 2-asset frontiers")
    weights = [np.array([w,1-w]) for w in np.linspace(0,1, n_points)]
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({
        "Returns": rets,
        "Volatility": vols
    })
    return ef.plot.line(x="Volatility", y="Returns", style=".-")



def minimize_volatility(target_return, er, cov):
    """
    Gives the weights for the given target return
    """
    num_assets = er.shape[0]
    # You have to give an objective function, some constrains and an initial guess to the optimizer.
    initial_guess = np.repeat(1 / num_assets, num_assets)
    # First constraint is, each weight should have some bounds, weights above 1 will be
    # equivalent of leverage, and negative weights will be equivalent of going short
    bounds = ((0.0, 1.0),) * num_assets  # for the optimizer, you have to give a series of bounds
    # We have to make sure that the return that we generate from that set of weights is the return of the target
    return_is_target = {
        'type': 'eq',  # equality constraint
        'args': (er,),
        'fun': lambda weights, er: target_return - portfolio_return(weights, er)
    }
    # The next constraint is the weights should be equal to 1
    weights_sum_to_1 = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1
    }
    # Use the quadratic programming optimizer which is called SLSQP
    results = minimize(portfolio_vol, initial_guess, args=(cov,), method="SLSQP",
                       options={'disp': False}, constraints=(return_is_target, weights_sum_to_1),
                       bounds=bounds)
    return results.x


def max_Sharpe_ratio(riskfree_rate, er, cov):
    """
    riskfree_rate + er + cov --> W
    """
    num_assets = er.shape[0]
    # You have to give an objective function, some constrains and an initial guess to the optimizer.
    initial_guess = np.repeat(1 / num_assets, num_assets)
    # First constraint is, each weight should have some bounds, weights above 1 will be
    # equivalent of leverage, and negative weights will be equivalent of going short
    bounds = ((0.0, 1.0),) * num_assets  # for the optimizer, you have to give a series of bounds

    # The next constraint is the weights should be equal to 1
    weights_sum_to_1 = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1
    }
    def neg_Sharpe_ratio(weights, riskfree_rate, er, cov):
        """
        Returns the negative of Sharpe ratio, given weights
        """
        returns = portfolio_return(weights, er)
        volatility = portfolio_vol(weights, cov)
        return -(returns - riskfree_rate)/volatility
    # Use the quadratic programming optimizer which is called SLSQP
    results = minimize(neg_Sharpe_ratio, initial_guess, args=(riskfree_rate, er, cov), method="SLSQP",
                       options={'disp': False}, constraints=(weights_sum_to_1),
                       bounds=bounds)
    return results.x


def optimal_weights(n_points, er, cov):
    """
    Generate list of weights to run the optimizer on to minimize the volatility
    We have to generate a list of target returns and send that to the optimizer to generate weights.
    Set of target returns can be derived using linearly spaced points between er.min (portfolio with lowest return)
    and er.max (portfolio with highest return)
    """
    target_returns = np.linspace(er.min(), er.max(), n_points)
    weights = [minimize_volatility(target_return, er, cov) for target_return in target_returns]
    return weights

def gmv(cov):
    """
    Returns the weight of the Global Minimum Volatility portfolio given covariance matrix
    """
    number_of_assets = cov.shape[0]
    #riskfree rate is not going to get used. For expected return, it really doesnt matter what number you give.
    return max_Sharpe_ratio(0, np.repeat(1, number_of_assets), cov)




def plot_ef(n_points, er, cov, show_cml=False, style= '.-', risk_free_rate=0, show_ew=False, show_gmv=False):
    """
    Plots the N Asset Efficient Frontier
    :param n_points: number of points
    :param er: expected returns
    :param cov: covariance matrix
    :return: plot of efficient frontier
    """
    # Have to find the minimum volatility for a certain return
    weights = optimal_weights(n_points, er, cov)
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({
        "Returns": rets,
        "Volatility": vols
    })
    ax = ef.plot.line(x="Volatility", y="Returns", style=".-")


    if show_gmv:
        weights_gmv = gmv(cov)
        returns_gmv = portfolio_return(weights_gmv, er)
        volatility_gmv = portfolio_vol(weights_gmv , cov)
        #display EW
        ax.plot([volatility_gmv], [returns_gmv], color="midnightblue", marker="o", markersize=10)

    if show_ew:
        num_of_assets = er.shape[0]
        weights_ew = np.repeat(1/num_of_assets, num_of_assets)
        returns_ew = portfolio_return(weights_ew, er)
        volatility_ew = portfolio_vol(weights_ew , cov)
        #display EW
        ax.plot([volatility_ew], [returns_ew], color="goldenrod", marker="o", markersize=12)

    if show_cml:
        ax.set_xlim(left=0)
        rf = 0.1
        weights_MSR = max_Sharpe_ratio(risk_free_rate, er, cov)
        returns_MSR = portfolio_return(weights_MSR, er)
        volatility_MSR = portfolio_vol(weights_MSR, cov)
        # Add CML
        cml_x = [0, volatility_MSR]
        cml_y = [rf, returns_MSR]
        ax.plot(cml_x, cml_y, color="green", marker="o", linestyle="dashed", markersize=12, linewidth=2)

    return ax



def run_cppi(risky_r, safe_r=None, m=3, start=1000, floor=0.8, riskfree_rate=0.03, drawdown=None):
    """
    Run a backtest of the CPPI strategy, given a set of returns for the risky asset.
    Returns a dictionary containing: Asset Value History, Risk Budget History, Risky Weight History
    """
    # Set up CPPI parameters
    dates = risky_r.index
    n_steps = len(dates)  # number of steps is the length of array
    account_value = start
    previous_peak = start
    # To compute the cushion, we need to know the floor value.
    floor_value = start * floor  # Initialize floor value

    if isinstance(risky_r, pd.Series):
        risky_r = pd.DataFrame(risky_r, columns=["R"])

    if safe_r is None:
        safe_r = pd.DataFrame().reindex_like(risky_r)
        safe_r.values[:] = riskfree_rate/12 # fast way to set all values to a number

    # Each stage of back test, we want to look at some variables (not necessary for CPPI algorithm)
    account_history = pd.DataFrame().reindex_like(risky_r)
    cushion_history = pd.DataFrame().reindex_like(risky_r)
    risky_w_history = pd.DataFrame().reindex_like(risky_r)  # Risky allocation weight history

    for step in range(n_steps):
        if drawdown is not None:
            previous_peak = np.maximum(previous_peak, account_value)
            floor_value = previous_peak * (1-drawdown)
        cushion = (account_value - floor_value) / account_value  # divide by account value to get percentage
        risky_w = m * cushion
        # What happens if account_value is 1000 and floor is 600? The cushion is 40%. If the multiplier is 3,
        # 40% multiplied by 3 is 120%. That means you have to borrow money, you are levering your investment.
        # We do not want to do that. We want to limit the allocation to the risky asset to be no more than a 100%.
        risky_w = np.minimum(risky_w, 1)
        risky_w = np.maximum(risky_w, 0)  # SHould be a positive number
        safe_w = 1 - risky_w
        # Now we know, what fraction of account value (dollars) we are going to allocate in safe asset and risky asset.
        # Now, in order to compute what the returns would be, you need to know how many dollars you are going to
        # allocate
        # to the risky asset and how many dollars you are going to allocate to safe asset.
        risky_allocation = account_value * risky_w
        safe_allocation = account_value * safe_w
        # Update the account value for the time step.
        account_value = risky_allocation * (1 + risky_r.iloc[step]) + safe_allocation * (1 + safe_r.iloc[step])
        # Save the values to look at the history and plot it
        cushion_history.iloc[step] = cushion
        risky_w_history.iloc[step] = risky_w
        account_history.iloc[step] = account_value

    # Compute risky wealth for comparison. Risky wealth is what you would have got if you had not done any CPPI.
    risky_wealth = start * (1 + risky_r).cumprod()  # compound the dollar and multiply by start value
    backtest_result = {
        "Wealth": account_history,
        "Risky Wealth": risky_wealth,
        "Risk_Budget": cushion_history,
        "Risky Allocation": risky_w_history,
        "m": m,
        "start": start,
        "floor": floor,
        "risky_r": risky_r,
        "safe_r": safe_r
    }
    return backtest_result


def get_total_market_index_returns():
    """
    Load the 30 industry portfolio data and derive the returns of a capweighted total market index
    """
    ind_nfirms = get_ind_nfirms()
    ind_size = get_ind_size()
    ind_return = get_ind_returns()
    ind_mktcap = ind_nfirms * ind_size
    total_mktcap = ind_mktcap.sum(axis=1)
    ind_capweight = ind_mktcap.divide(total_mktcap, axis="rows")
    total_market_return = (ind_capweight * ind_return).sum(axis="columns")
    return total_market_return


def summary_stats(r, riskfree_rate=0.03):
    """
    Return a DataFrame that contains aggregated summary stats for the returns in the columns of r
    """
    ann_r = r.aggregate(annualize_rets, periods_per_year=12)
    ann_vol = r.aggregate(annualize_vol, periods_per_year=12)
    ann_sr = r.aggregate(sharpe_ratio, riskfree_rate=riskfree_rate, periods_per_year=12)
    dd = r.aggregate(lambda r: drawdown(r).Drawdowns.min())
    skew = r.aggregate(skewness)
    kurt = r.aggregate(kurtosis)
    cf_var5 = r.aggregate(var_gaussian, modified=True)
    hist_cvar5 = r.aggregate(cvar_historic)
    return pd.DataFrame({
        "Annualied Return": ann_r,
        "Annualized Vol": ann_vol,
        "Skewness": skew,
        "Kurtosis": kurt,
        "Cornish-Fisher VaR (5%)": cf_var5,
        "Historic CVaR (5%)": hist_cvar5,
        "Sharpe Ratio": ann_sr,
        "Max Drawdown": dd
    })


def gbm(n_years = 10, n_scenarios = 1000, mu =0.07, sigma=0.15, steps_per_year=12, s_0=100, prices=True):
    """
         Evolution of a stock price using a Geometric Brownian Motion Model through Monte Carlo
         n_years worth of returns
         n_scenarios --> different versions
         dt --> micro step, move time forward, generate a stock return.
         s_0 --> initial stock price
         xi is a random normal
    """
    dt = 1/steps_per_year   # year divided by steps
    n_steps = int(n_years*steps_per_year) + 1
    returns_plus_1 = np.random.normal(loc=(1+mu*dt), scale=(sigma*np.sqrt(dt)), size=(n_steps, n_scenarios))
    returns_plus_1[0] = 1 # the return is zero
    ret_val = s_0 * pd.DataFrame(returns_plus_1).cumprod() if prices else returns_plus_1-1
    return ret_val


# def pv(l, r):
#     """
#     Computes the present value of a sequence of liabilities
#     l is indexed by the time, and the values are the amounts of each liability returns the present value of the
#     sequence.
#     """

#     dates = l.index
#     discounts = discount(dates, r)
#     return (discounts * l).sum()
def pv(flows, r):
    """
    Compute the present value of a sequence of cash flows given by the time (as an index) 
    and amounts r can be a scalar, or a Series or DataFrame with the number of rows matching
    the number of rows in flows. 
    """
    dates = flows.index
    discounts = discount(dates, r)
    return discounts.multiply(flows, axis='rows').sum()


# def discount(t, r):
#     """
#     Compute the price of a pure discount bond that pays a dollar at time t, given interest rate r
#     """
#     return (1+r)**(-t)
def discount(t, r):
    """
    Compute the price of a pure discount bond that pays a dollar at time period t 
    and r is the per-period interest rate.
    Returns a |t|*|r| Series or DataFrame.
    r can be a float, Series or DataFrame.
    Returns a DataFrame indexed by t
    """
    discounts = pd.DataFrame([(r+1)**-i for i in t])
    discounts.index = t
    return discounts


def funding_ratio(assets, liabilities, r):
    """
    Computes the funding ratio of a series of liabilities, based on an interest rate 
    and current value of assets.
    """
    return pv(assets, r)/pv(liabilities, r)


def inst_to_ann(r):
    """
    Converts short rate to an annualized rate.
    """
    return np.expm1(r)


def ann_to_inst(r):
    """
    Convert annualized to short rate.
    """
    return np.log1p(r)


def cir(n_years=10, n_scenarios=1, a=0.05, b=0.03, sigma=0.05, steps_per_year=12, r_0=None):
    """
    Generate random interest rate evolution over time using the CIR model. b and r_0 are assumed to be 
    the annualized rates, not the short rate, The returned values are the annualized rates as well.
    """
    if r_0 is None: r_0=b
    r_0 = ann_to_inst(r_0)
    dt = 1/steps_per_year
    num_steps = int(n_years*steps_per_year) + 1 # because n_years might be a float.
    
    shock = np.random.normal(0, scale=np.sqrt(dt), size=(num_steps, n_scenarios))
    rates = np.empty_like(shock)
    rates[0] = r_0
    
    ## For Price Generation
    h = math.sqrt(a**2 + 2*sigma**2)
    prices = np.empty_like(shock)
    
    def price(ttm, r):
        _A = ((2*h*math.exp((h+a)*ttm/2))/(2*h+(h+a)*(math.exp(h*ttm)-1)))**(2*a*b/sigma**2)
        _B = (2*(math.exp(h*ttm)-1))/(2*h + (h+a)*(math.exp(h*ttm)-1))
        _P = _A*np.exp(-_B*r)
        return _P
    
    prices[0] = price(n_years, r_0)
    
    for step in range(1, num_steps):
        r_t = rates[step-1]
        d_r_t = a*(b-r_t)*dt + sigma*np.sqrt(r_t)*shock[step]
        rates[step] = abs(r_t + d_r_t)
        # generate prices at time t as well
        prices[step] = price(n_years-step*dt, rates[step])
    
    rates = pd.DataFrame(data=inst_to_ann(rates), index=range(num_steps))
    ## For prices
    prices = pd.DataFrame(data=prices, index=range(num_steps))
    
    return rates, prices
    
      
        
def show_cir_prices(r_0=0.03, a=0.5, b=0.03, sigma=0.05, n_scenarios=5):
    cir(r_0=r_0, a=a, b=b, sigma=sigma, n_scenarios=n_scenarios)[1].plot(legend=False, figsize=(12,5))
    controls = widgets.interactive(show_cir_prices, r_0=(0, .15, .01), a=(0, 1, .1), b=(0, .15, .01), 
                               sigma=(0, .1, .01), n_scenarios=(1,100))
    display(controls)      
    return



def bond_cash_flows(maturity, principal=100, coupon_rate=0.03, coupons_per_year=12):
    """
    Returns a series of cash flows generated by a bond, indexed by a coupon number.
    """
    n_coupons = round(maturity*coupons_per_year)
    coupon_amt = principal*coupon_rate/coupons_per_year
    coupon_times = np.arange(1, n_coupons+1)
    cash_flows = pd.Series(data = coupon_amt, index=coupon_times)
    cash_flows.iloc[-1] += principal
    return cash_flows


# def bond_price(maturity, pricipal=100, coupon_rate=0.03, coupons_per_year=12, discount_rate=0.03):
#     """
#     Price a bond based on bond parameters maturity, principal, coupon rate and coupons_per_year 
#     and the prevailing discount rate. 
#     """
#     cash_flows = bond_cash_flows(maturity, pricipal, coupon_rate, coupons_per_year)
#     return pv(cash_flows, discount_rate/coupons_per_year)

# Include to process discount rate as a dataframe in the bond price function.
def bond_price(maturity, principal=100, coupon_rate=0.03, coupons_per_year=12, discount_rate=0.03):
    """
    Computes the price of a bond that pays regular coupons until maturity at which time 
    the principal and the final coupon is returned.
    This is not designed to be efficient, rather, it is to illustrate the underlying 
    principle behind bond pricing.
    If discount_rate is a DataFrame, then this is assumed to be the rate on each coupon rate and 
    the bond value is computed over time.
    i.e: The index of the discount_rate DataFrame is assumed to be the coupon number.
    """
    if isinstance(discount_rate, pd.DataFrame):
        pricing_dates = discount_rate.index
        prices = pd.DataFrame(index=pricing_dates, columns=discount_rate.columns)
        for t in pricing_dates:
            prices.iloc[t] = bond_price(maturity-t/coupons_per_year, principal, coupon_rate, 
                                        coupons_per_year, discount_rate.loc[t])
        return prices
    else: # base case ...single time period
        if maturity <= 0: return principal+principal*coupon_rate/coupons_per_year
        cash_flows = bond_cash_flows(maturity, principal, coupon_rate, coupons_per_year)
        return pv(cash_flows, discount_rate/coupons_per_year)
    
    
def bond_total_return(monthly_prices, principal, coupon_rate, coupons_per_year):
    """
    Computes the total return of a bond based on monthly bond prices and coupon payments. 
    Assumes that dividends (coupons) are paid out at the end of the period (eg: end of 
    3 months for quarterly div) and that dividends are reinvested in bonds. 
    """
    coupons = pd.DataFrame(data=0, index=monthly_prices.index, columns=monthly_prices.columns)
    t_max = monthly_prices.index.max()
    # Space pay dates over time - monthly 
    pay_date = np.linspace(12/coupons_per_year, t_max, int(coupons_per_year*t_max/12), dtype=int)
    coupons.iloc[pay_date] = principal*coupon_rate/coupons_per_year
    total_returns = (monthly_prices + coupons)/monthly_prices.shift()-1
    return total_returns.dropna()



def macaulay_duration(cash_flows, discount_rate):
    """
    Computes the Macaulay Duration of a sequence of cash flows.
    """
    discounted_flows = discount(cash_flows.index, discount_rate) * cash_flows
    weights = discounted_flows/discounted_flows.sum()
    return np.average(cash_flows.index, weights=weights)



def match_durations(cf_t, cf_s, cf_l, discount_rate):
    """
    Returns the weight W in cf_s that, along with (1-W) in cf_l will have an effective duration
    that matches cf_t
    """
    d_t = macaulay_duration(cf_t, discount_rate)
    d_s = macaulay_duration(cf_s, discount_rate)
    d_l = macaulay_duration(cf_l, discount_rate)
    return (d_l-d_t)/(d_l-d_s)



# Backtester to test different mixes between two different portfolios from returns
def backtest_mix(r1, r2, allocator, **kwargs):
    """
    Runs a back test (simulation) of allocating between two sets of returns.
    r1 and r2 are T*N DataFrames or returns where T is the time step index and 
    N is the number of scenarios.
    allocator is a function that takes two sets of returns and allocator specific parameters.
    and produces an allocation to the first portfolio (the rest of the money is invested 
    in the GHP) as a T*1 DataFrame.
    Returns a T*N DataFrame of the resulting N portfolio scenarios.
    """
    
    if not r1.shape == r2.shape:
        raise ValueError("r1 and r2 need to be the same shape")
        
    weights = allocator(r1, r2, **kwargs)
    
    if not weights.shape == r1.shape:
        raise ValueError("Allocator returned weights that don't match r1")
        
    r_mix = weights*r1 + (1-weights)*r2
    return r_mix
    
    
    
# Write a simpe allocator
def fixed_mix_allocator(r1, r2, w1, **kwargs):
    """
    Produces a time series over T steps of allocations between the PSP and GHP across N scenarios.
    PSP and GHP are T*N DataFrames that represent the returns of the PSP and GHP such that:
        - each column is a scenario
        - each row is the price for a timestep.
    Returns an T*N DataFrame of PSP weights.
    """
    # same mix every single time
    return pd.DataFrame(data=w1, index= r1.index, columns=r1.columns)



def terminal_values(rets):
    """
    Returns the final values (of a dollar) at the end of the return period for each scenario.
    """
    # It is nothing more than the compounded return.
    return (rets+1).prod()


def terminal_stats(rets, floor=0.8, cap=np.inf, name="Stats"):
    """
    Produce summary statistics on the terminal values per invested dollar across a range of N scenarios.
    rets is a T*N DataFrame of return, where T is the time-step (we assume returns are sorted by time).
    Returns a 1 column DataFrame of Summary Stats indexed by the stats name.
    """
    terminal_wealth = (rets+1).prod()
    breach = terminal_wealth < floor
    reach = terminal_wealth >= cap
    p_breach = breach.mean() if breach.sum() > 0 else np.nan
    p_reach = breach.mean() if reach.sum() > 0 else np.nan
    e_short = (floor - terminal_wealth[breach]).mean() if breach.sum() > 0 else np.nan
    e_surplus = (cap - terminal_wealth[reach]).mean() if reach.sum() > 0 else np.nan
    sum_stats = pd.DataFrame.from_dict({
        "mean": terminal_wealth.mean(),
        "std": terminal_wealth.std(),
        "p_breach": p_breach,
        "e_short": e_short,
        "p_reach": p_reach,
        "e_surplus": e_surplus
    }, orient="index", columns=[name])
    return sum_stats


def glidepath_allocator(r1, r2, start_glide=1, end_glide=0):
    """
    Simulates a Target-Date-Fund style gradual move from r1 to r2.
    """
    n_points = r1.shape[0]
    n_column = r1.shape[1]
    glide_path = pd.Series(data=np.linspace(start_glide, end_glide, num=n_points))
    # we need to return as many columns as r1
    paths = pd.concat([glide_path]*n_column, axis=1)
    paths.index = r1.index
    paths.columns = r1.columns
    return paths



# New allocator which takes consideration of the floor
def floor_allocator(psp_r, ghp_r, floor, zc_prices, m=3):
    """
    Allocate between PSP and GHP withthe goal to provide exposure to the upside
    of the PSP without violating the floor requirement.
    Uses a CPPI-style dynamic risk budgeting algorithm by investing a multiple of the cushion 
    in the PSP.
    Returns a dataframe with the same shape as the php/ghp representing the weights in the PSP.
    """
    if zc_prices.shape != psp_r.shape:
        raise ValueError("PSP and ZC prices must have the same shape")
    
    n_steps, n_scenarios = psp_r.shape
    # Start with a $1 in every single scenario
    account_value = np.repeat(1, n_scenarios)
    floor_value = np.repeat(1, n_scenarios)
    w_history = pd.DataFrame(index=psp_r.index, columns=psp_r.columns)
    for step in range(n_steps):
        floor_value = floor*zc_prices.iloc[step] # PV of Floor assuming today's rates and flat YC
        cushion = (account_value - floor_value)/account_value
        psp_w = (m*cushion).clip(0,1) #same as applying min and max
        ghp_w = 1 - psp_w
        psp_alloc = account_value*psp_w
        ghp_alloc = account_value*ghp_w
        # Recompute the account value at the end of this step.
        account_value = psp_alloc * (1+psp_r.iloc[step]) + ghp_alloc * (1+ghp_r.iloc[step])
        w_history.iloc[step] = psp_w
    return w_history



def drawdown_allocator(psp_r, ghp_r, maxdd, m=3):
    """
    Allocate between PSP and GHP withthe goal to provide exposure to the upside
    of the PSP without violating the floor requirement.
    Uses a CPPI-style dynamic risk budgeting algorithm by investing a multiple of the cushion 
    in the PSP.
    Returns a dataframe with the same shape as the php/ghp representing the weights in the PSP.
    """
    n_steps, n_scenarios = psp_r.shape
    # Start with a $1 in every single scenario
    account_value = np.repeat(1, n_scenarios)
    floor_value = np.repeat(1, n_scenarios)
    peak_value = np.repeat(1, n_scenarios)
    w_history = pd.DataFrame(index=psp_r.index, columns=psp_r.columns)
    for step in range(n_steps):
        floor_value = (1-maxdd) * peak_value # Floor is based on previous peak
        cushion = (account_value - floor_value)/account_value
        psp_w = (m*cushion).clip(0,1) #same as applying min and max
        ghp_w = 1 - psp_w
        psp_alloc = account_value*psp_w
        ghp_alloc = account_value*ghp_w
        # Recompute the new account value and prev peak at the end of this step.
        account_value = psp_alloc * (1+psp_r.iloc[step]) + ghp_alloc * (1+ghp_r.iloc[step])
        peak_value = np.maximum(peak_value, account_value)
        w_history.iloc[step] = psp_w
    return w_history


def regress(dependent_variable, explanatory_variables, alpha=True):
    """
    Runs a linear regression to decompose the dependent variable into the explanatory variables
    returns an object of type statsmodel's RegressionResults on which you can call
       .summary() to print a full summary
       .params for the coefficients
       .tvalues and .pvalues for the significance levels
       .rsquared_adj and .rsquared for quality of fit
    """
    if alpha:
        explanatory_variables = explanatory_variables.copy()
        explanatory_variables["Alpha"] = 1

    lm = sm.OLS(dependent_variable, explanatory_variables).fit()
    return lm


def style_analysis(dependant_variables, explanatory_variables):
    """
    Returns the optimal weights that minimizes the Tracking error
    between a portfolio of the explanatory variables and the dependant variables.
    Code is slight variation to the minimize_vol function in toolkit.
    """
    n = explanatory_variables.shape[1]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n # an N-tuple of 2-tuples.
    # Constraints
    weights_sum_to_1 = {
                        'type':'eq',
                        'fun': lambda weights: np.sum(weights) - 1
                        }
    solution = minimize(portfolio_tracking_error,
                        init_guess,
                        args = (dependant_variables, explanatory_variables,),
                        method = 'SLSQP',
                        options = {'disp':False},
                        constraints = (weights_sum_to_1,),
                        bounds = bounds
                       )
    weights = pd.Series(solution.x, index=explanatory_variables.columns)
    return weights


def tracking_error(r_a, r_b):
    """
    Returns the Tracking Error between the two return series
    """
    return np.sqrt(((r_a - r_b)**2).sum())


def portfolio_tracking_error(weights, reference_returns, buildingblock_returns):
    """
    Returns the tracking error between the reference returns
    and a portfolio of building block returns held with given weights.
    """
    return tracking_error(reference_returns, (weights*buildingblock_returns).sum(axis=1))


def weight_ew(r, cap_weights=None, max_cw_mult=None, microcap_threshold=None, **kwargs):
    """
    Returns the weights of the EW portfolio based on the asset returns "r" as a DataFrame
    If supplied a set of capweights and a capweight tether, it is applied and reweighted
    """
    n = len(r.columns)
    ew = pd.Series(1/n, index=r.columns)
    if cap_weights is not None:
        cw = cap_weights.loc[r.index[0]] # starting cap weight
        ## exclude microcaps
        if microcap_threshold is not None and microcap_threshold > 0:
            microcap = cw < microcap_threshold
            ew[microcap] = 0
            ew = ew/ew.sum()
        #limit weight to a multiple of capweight
        if max_cw_mult is not None and max_cw_mult > 0:
            ew = np.minimum(ew, cw*max_cw_mult)
            ew = ew/ew.sum() #reweight
    return ew

def weight_cw(r, cap_weights, **kwargs):
    """
    Returns the weights of the CW portfolio based on the time series of capweights
    """
    w = cap_weights.loc[r.index[1]]
    return w/w.sum()

def backtest_ws(r, estimation_window=60, weighting=weight_ew, **kwargs):
    """
    Backtests a given weighting scheme, given some parameters:
    r : asset returns to use to build the portfolio
    estimation_window: the window to use to estimate parameters
    weighting: the weighting scheme to use, must be a function that takes "r", and a variable number of keyword-value arguments
    """
    n_periods = r.shape[0]
    # return windows
    windows = [(start, start+estimation_window) for start in range(n_periods-estimation_window+1)]
    weights = [weighting(r.iloc[win[0]:win[1]], **kwargs) for win in windows]
    # convert to DataFrame
    weights = pd.DataFrame(weights, index=r.iloc[estimation_window-1:].index, columns=r.columns)
    returns = (weights * r).sum(axis="columns",  min_count=1) #mincount is to generate NAs if all inputs are NAs
    return returns

def sample_cov(r, **kwargs):
    """
    Returns the sample covariance of the supplied returns.
    """
    return r.cov()


def weight_gmv(r, cov_estimator=sample_cov, **kwargs):
    """
    Produces the weights of the GMV portfolio given a covariance matix of the returns.
    """
    est_cov = cov_estimator(r, **kwargs)
    return gmv(est_cov)


def constant_correlation_cov(r, **kwargs):
    """
    Estimates a covariance matrix by using the Elton/Gruber Constant Correlation Model.
    """

    rhos = r.corr()
    n = rhos.shape[0]
    # This is a symmetric matrix with all diagonals = 1
    # So the mean correlation matri is the sum of all elements minus the number of elements in diagonal
    # [average of off-diagonal elements]
    rho_bar = (rhos.values.sum() - n) / (n * (n - 1))
    # Reconstruct a new crrelation matrix with the mean values
    ccor = np.full_like(rhos, rho_bar)
    # Diagonal values should be 1
    np.fill_diagonal(ccor, 1)
    sd = r.std()

    ## regenerate the covariance matrix. There are two ways of doing it, one with the statsmodels package
    #     import statsmodels.stats.moment_helpers as mh
    #     ccov = mh.corr2cov(ccor, sd)
    ccov = ccor * np.outer(sd, sd)

    return pd.DataFrame(ccov, index=r.columns, columns=r.columns)



def shrinkage_cov(r, delta=0.5, **kwargs):
    """
    Covariance estimator that shrinks between the Sample Covariance and the Constant Correlation Estimators.
    """
    prior = constant_correlation_cov(r, **kwargs)
    sample = sample_cov(r, **kwargs)
    return delta*prior + (1-delta)*sample



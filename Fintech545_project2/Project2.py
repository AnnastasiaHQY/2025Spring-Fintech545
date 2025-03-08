################################################################################
# PROBLEM 1: PORTFOLIO RISK ANALYSIS
################################################################################

import pandas as pd
import numpy as np
from scipy.stats import norm, t
import matplotlib.pyplot as plt

# Part A: Calculate Arithmetic and Log Returns
# ----------------------------------------------------------------------------

# Read data
df = pd.read_csv("DailyPrices.csv", parse_dates=['Date'])

selected_stocks = ['SPY', 'AAPL', 'EQIX']
filtered_df = df[['Date'] + selected_stocks].set_index('Date')

# A. Calculate Arithmetic Returns
arithmetic_return = filtered_df.pct_change().dropna()
arithmetic_return_remove_mean = arithmetic_return - arithmetic_return.mean()

print('A. Arithmetic Returns')
print('Last five rows:')
print(arithmetic_return_remove_mean.tail())
print('\n' + 'Standard deviations:')
print(arithmetic_return_remove_mean.std())

# B. Calculate Log Returns
log_return = np.log(filtered_df).diff().dropna()
log_return_remove_mean = log_return - log_return.mean()

print('\nB. Log Returns')
print('Last five rows:')
print(log_return_remove_mean.tail())
print('\n' + 'Standard deviations:')
print(log_return_remove_mean.std())

# Part B: Value at Risk (VaR) and Expected Shortfall (ES) Analysis
# ----------------------------------------------------------------------------

# Load the daily prices data
df = pd.read_csv('DailyPrices.csv', parse_dates=['Date'])
df.sort_values('Date', inplace=True)
df.set_index('Date', inplace=True)

# Get the prices on 2025-01-03 (if not available, asof will return the last available price)
prices_on_date = df.asof('2025-01-03')

# Define the portfolio positions
portfolio_positions = {'SPY': 100, 'AAPL': 200, 'EQIX': 150}

# Calculate the dollar value of each position and the overall portfolio value
position_values = {asset: portfolio_positions[asset] * prices_on_date[asset] for asset in portfolio_positions}
total_portfolio_value = sum(position_values.values())
print(f"Portfolio Value on 2025-01-03: ${total_portfolio_value:.2f}")

# Calculate daily arithmetic returns and center them (zero mean)
daily_returns = df.pct_change().dropna()
# Center returns to have zero mean
centered_returns = daily_returns - daily_returns.mean()

# VaR and ES at the 5% level using three different methods
alpha = 0.05
assets = list(portfolio_positions.keys())

# Method A: Normal Distribution with Exponentially Weighted Covariance (λ = 0.97)
# ----------------------------------------------------------------------------

lambda_decay = 0.97
num_obs = len(centered_returns)
# Create exponential weights giving more weight to recent observations
exp_weights = np.array([(1 - lambda_decay) * (lambda_decay ** (num_obs - i - 1)) for i in range(num_obs)])
exp_weights /= exp_weights.sum()

# Calculate asset volatilities using the exponentially weighted variance
asset_vols = {}
for asset in assets:
    variance = np.sum(exp_weights * (centered_returns[asset] ** 2))
    asset_vols[asset] = np.sqrt(variance)

# Compute the exponentially weighted covariance matrix
cov_matrix = np.zeros((len(assets), len(assets)))
for i, asset_i in enumerate(assets):
    for j, asset_j in enumerate(assets):
        cov_matrix[i, j] = np.sum(exp_weights * centered_returns[asset_i] * centered_returns[asset_j])

# Compute the dollar-weighted portfolio weights
portfolio_weights_dollar = np.array([position_values[asset] for asset in assets]) / total_portfolio_value
portfolio_vol = np.sqrt(portfolio_weights_dollar.T @ cov_matrix @ portfolio_weights_dollar)

# Calculate VaR and ES analytically under the normal assumption
z_score = norm.ppf(alpha)  # this will be negative for alpha=0.05
var_normal_ind = {}
es_normal_ind = {}
for asset in assets:
    var_normal_ind[asset] = -z_score * asset_vols[asset] * position_values[asset]
    es_normal_ind[asset] = (asset_vols[asset] * norm.pdf(z_score) / alpha) * position_values[asset]

var_normal_portfolio = -z_score * portfolio_vol * total_portfolio_value
es_normal_portfolio = (portfolio_vol * norm.pdf(z_score) / alpha) * total_portfolio_value

print("\nMethod A: Normal Distribution with Exponentially Weighted Covariance")
for asset in assets:
    print(f"{asset}: VaR = ${var_normal_ind[asset]:.2f}, ES = ${es_normal_ind[asset]:.2f}")
print(f"Portfolio: VaR = ${var_normal_portfolio:.2f}, ES = ${es_normal_portfolio:.2f}")


# Method B: T-distribution with Gaussian Copula
# ----------------------------------------------------------------------------

def simulate_t_copula_var_es(returns_df, asset_weights, portfolio_val, significance=0.05, num_sim=10000):
    """
    Simulate portfolio VaR and ES using a t-distribution based Gaussian copula.

    Parameters:
      returns_df   : DataFrame of centered returns.
      asset_weights: Array of asset weights (in dollar terms as a fraction of total portfolio).
      portfolio_val: Total portfolio value.
      significance : VaR significance level (default 5%).
      num_sim      : Number of simulations.

    Returns:
      Tuple of (VaR, ES) for the portfolio.
    """
    # Ensure the returns are centered
    centered_df = returns_df - returns_df.mean()
    asset_list = centered_df.columns

    # Fit a t-distribution to each asset and store parameters in a dictionary.
    t_params = {asset: t.fit(centered_df[asset]) for asset in asset_list}

    # Convert the centered returns to uniform variables using the fitted t CDF.
    uniform_df = pd.DataFrame(index=centered_df.index, columns=centered_df.columns)
    for asset in asset_list:
        df_param, loc, scale = t_params[asset]
        uniform_df[asset] = t.cdf(centered_df[asset], df_param, loc, scale)

    # Map uniform variables to standard normals using the inverse CDF (probit function).
    normal_mapped = uniform_df.apply(lambda col: norm.ppf(col))

    # Calculate the correlation matrix from the transformed data.
    corr_matrix = normal_mapped.corr().values

    # Generate correlated standard normal random variables.
    np.random.seed(42)
    correlated_normals = np.random.multivariate_normal(mean=np.zeros(len(asset_list)), cov=corr_matrix, size=num_sim)

    # Transform the correlated normals back into uniform samples.
    uniform_samples = norm.cdf(correlated_normals)

    # Transform the uniform samples into t-distributed samples using the inverse t CDF.
    simulated_returns = np.empty_like(uniform_samples)
    for idx, asset in enumerate(asset_list):
        df_param, loc, scale = t_params[asset]
        simulated_returns[:, idx] = t.ppf(uniform_samples[:, idx], df_param, loc, scale)

    # Compute the simulated portfolio returns (as a weighted sum across assets)
    sim_portfolio_returns = simulated_returns.dot(asset_weights)
    sorted_returns = np.sort(sim_portfolio_returns)

    cutoff = int(significance * num_sim)
    var_sim = -sorted_returns[cutoff] * portfolio_val
    es_sim = -np.mean(sorted_returns[:cutoff]) * portfolio_val

    return var_sim, es_sim


# Use the function to compute portfolio-level risk metrics
var_tcopula_port, es_tcopula_port = simulate_t_copula_var_es(centered_returns[assets],
                                                             portfolio_weights_dollar,
                                                             total_portfolio_value,
                                                             significance=alpha,
                                                             num_sim=10000)
print("\nMethod B: T-distribution with Gaussian Copula (Portfolio Level)")
print(f"Portfolio: VaR = ${var_tcopula_port:.2f}, ES = ${es_tcopula_port:.2f}")


# For individual assets, we can simulate separately.
def simulate_t_copula_individual(return_series, pos_value, significance=0.05, num_sim=10000):
    centered_series = return_series - return_series.mean()
    # Fit t-distribution parameters for the asset.
    params = t.fit(centered_series)
    np.random.seed(42)
    sim_returns = t.rvs(*params, size=num_sim)
    sorted_sim = np.sort(sim_returns)
    cutoff_index = int(significance * num_sim)
    var_asset = -sorted_sim[cutoff_index] * pos_value
    es_asset = -np.mean(sorted_sim[:cutoff_index]) * pos_value
    return var_asset, es_asset


var_tcopula_indiv = {}
es_tcopula_indiv = {}
for asset in assets:
    v, e = simulate_t_copula_individual(centered_returns[asset], position_values[asset], significance=alpha,
                                        num_sim=10000)
    var_tcopula_indiv[asset] = v
    es_tcopula_indiv[asset] = e

for asset in assets:
    print(f"{asset}: VaR = ${var_tcopula_indiv[asset]:.2f}, ES = ${es_tcopula_indiv[asset]:.2f}")


# Method C: Historical Simulation
# ----------------------------------------------------------------------------

def historical_simulation_var_es(returns_df, position_dict, significance=0.05):
    """
    Calculate historical VaR and ES for the portfolio using past data.

    Parameters:
      returns_df  : DataFrame of centered returns.
      position_dict: Dictionary mapping each asset to its dollar position.
      significance: VaR significance level.

    Returns:
      Tuple of (VaR, ES) for the portfolio.
    """
    # Calculate daily dollar changes for each asset.
    daily_pnl = pd.DataFrame()
    for asset in returns_df.columns:
        daily_pnl[asset] = returns_df[asset] * position_dict[asset]

    # Compute the overall portfolio daily profit and loss.
    portfolio_pnl = daily_pnl.sum(axis=1)

    # Compute the VaR as the quantile and the ES as the mean loss below that quantile.
    var_hist = np.quantile(portfolio_pnl, significance)
    es_hist = portfolio_pnl[portfolio_pnl <= var_hist].mean()

    return -var_hist, -es_hist


var_hist_port, es_hist_port = historical_simulation_var_es(centered_returns[assets], position_values,
                                                           significance=alpha)
print("\nMethod C: Historical Simulation (Portfolio Level)")
print(f"Portfolio: VaR = ${var_hist_port:.2f}, ES = ${es_hist_port:.2f}")


# For individual assets using historical simulation:
def historical_simulation_individual(return_series, pos_value, significance=0.05):
    pnl = return_series * pos_value
    var_ind = np.quantile(pnl, significance)
    es_ind = pnl[pnl <= var_ind].mean()
    return -var_ind, -es_ind


var_hist_indiv = {}
es_hist_indiv = {}
for asset in assets:
    v, e = historical_simulation_individual(centered_returns[asset], position_values[asset], significance=alpha)
    var_hist_indiv[asset] = v
    es_hist_indiv[asset] = e

for asset in assets:
    print(f"{asset}: VaR = ${var_hist_indiv[asset]:.2f}, ES = ${es_hist_indiv[asset]:.2f}")

# Part C: Discussion on the Differences between the Methods
# ----------------------------------------------------------------------------

discussion = """
Discussion:
- Method A (Normal Distribution) has the lowest portfolio VaR and ES estimates. This approach assumes returns follow a normal distribution with exponentially weighted covariance. It gives more weight to recent observations. It is easy to calculate but often underestimates tail risk because financial returns usually have fatter tails than the normal distribution.
- Method B (T-Distribution with Gaussian Copula) gives VaR and ES values that are in between the other two methods. This method captures the heavy tails of asset returns better by using the t-distribution. It uses a Gaussian copula to model how assets depend on each other. The higher ES values compared to Method A show that it does a better job of accounting for large losses.
- Method C (Historical Simulation) gives the highest VaR estimates but slightly lower ES than Method B. This method does not assume any distribution. Instead, it uses historical returns directly. It shows real market behaviors, including extreme events from the historical data. However, it depends heavily on the historical period chosen.

"""
print(discussion)


################################################################################
# PROBLEM 3: OPTIONS PRICING AND RISK ANALYSIS
################################################################################

import numpy as np
import math
import scipy.stats as st
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq

###############################################################################
# Black–Scholes Functions
###############################################################################
def black_scholes_call(S, K, r, T, sigma):
    """
    Returns the Black-Scholes price of a European Call.
    S:    Spot price
    K:    Strike price
    r:    Risk-free interest rate (annualized)
    T:    Time to maturity (in years)
    sigma:Volatility (annualized)
    """
    if T <= 0:
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    call_price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return call_price

def black_scholes_put(S, K, r, T, sigma):
    """
    Returns the Black-Scholes price of a European Put.
    """
    if T <= 0:
        return max(K - S, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    put_price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put_price

###############################################################################
# 1. Implied Volatility
###############################################################################
def implied_vol_call(market_price, S, K, r, T, lower=1e-6, upper=3.0):
    """
    Solve for implied volatility using a root finder so that
    black_scholes_call(...) = market_price.
    """
    def objective(sigma):
        return black_scholes_call(S, K, r, T, sigma) - market_price
    iv = brentq(objective, lower, upper, maxiter=200)
    return iv

###############################################################################
# 2. Greeks: Delta, Vega, Theta
###############################################################################
def bs_greeks_call(S, K, r, T, sigma):
    """
    Returns (Delta, Vega, Theta) for a European Call option (no dividends).
    """
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    delta = norm.cdf(d1)
    vega = S * norm.pdf(d1) * math.sqrt(T)
    theta = - (S * norm.pdf(d1) * sigma) / (2.0 * math.sqrt(T)) - r * K * math.exp(-r * T) * norm.cdf(d2)
    return delta, vega, theta

def bs_greeks_put(S, K, r, T, sigma):
    """
    Returns (Delta, Theta) for a European Put option.
    """
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    delta = norm.cdf(d1) - 1  # Put Delta
    theta = - (S * norm.pdf(d1) * sigma) / (2.0 * math.sqrt(T)) + r * K * math.exp(-r * T) * norm.cdf(-d2)
    return delta, theta

###############################################################################
# Main Parameters (from problem statement)
###############################################################################
S = 31.0      # Current stock price
K = 30.0      # Strike price
r = 0.10      # Risk-free rate (10%)
T = 0.25      # 3 months = 0.25 years
C_obs = 3.00  # Observed call premium

# A) Implied Volatility
iv = implied_vol_call(C_obs, S, K, r, T)
print(f"(A) Implied Volatility = {iv * 100:.2f}%")

# B) Greeks for the call & approximate price change if vol +1%
delta, vega, theta = bs_greeks_call(S, K, r, T, iv)
print(f"(B) Delta = {delta:.4f}, Vega = {vega:.4f}, Theta (annual) = {theta:.4f}")
new_iv = iv + 0.01
new_price_call = black_scholes_call(S, K, r, T, new_iv)
price_change = new_price_call - black_scholes_call(S, K, r, T, iv)
print(f"Actual price change if vol increases by +1% = {price_change:.4f}")
# C) Price the put using the Black-Scholes-Merton formula
# ----------------------------------------------------------------------------

put_price = black_scholes_put(S, K, r, T, iv)
print("\nC. Put Price: {:.4f}".format(put_price))

# Check Put-Call Parity: C - P should equal S - K*exp(-rT)
parity_diff = (black_scholes_call(S, K, r, T, iv) - put_price) - (S - K * math.exp(-r * T))
print("   Put-Call Parity difference: {:.4e}".format(parity_diff))

###############################################################################
# D) VaR and ES for the portfolio {1 Call, 1 Put, 1 Share} over 20 trading days
###############################################################################
# Portfolio today: 1 call + 1 put + 1 share
call_value = black_scholes_call(S, K, r, T, iv)
put_value = black_scholes_put(S, K, r, T, iv)
portfolio_value_0 = call_value + put_value + S
print("(D) Portfolio Composition (today):")
print(f"    Call Value = {call_value:.4f}")
print(f"    Put  Value = {put_value:.4f}")
print(f"    Stock      = {S:.4f}")
print(f"    => Portfolio Value = {portfolio_value_0:.4f}\n")

# Compute portfolio sensitivities (using call and put greeks)
delta_c, vega_c, theta_c = bs_greeks_call(S, K, r, T, iv)
delta_p, theta_p = bs_greeks_put(S, K, r, T, iv)
# For a stock, delta = 1 and theta = 0
portfolio_delta = delta_c + delta_p + 1.0
portfolio_theta = theta_c + theta_p
print(f"    Portfolio Delta = {portfolio_delta:.4f}")
print(f"    Portfolio Theta = {portfolio_theta:.4f}")

# Holding period parameters:
holding_days = 20
days_per_year = 255
dt = holding_days / days_per_year  # in years

# Stock volatility (annual) = 25%
annual_vol = 0.25
# Standard deviation of stock price change over dt:
std_S = S * annual_vol * math.sqrt(dt)

# ---------------------------
# (i) Delta–Normal Approximation
# Using the second method's idea for VaR calculation:

# Mean change due to time decay over holding period:
mean_change = portfolio_theta * dt
# Standard deviation of portfolio change due to delta exposure:
std_portfolio = portfolio_delta * std_S

# Using the 5% quantile (z_0.05) of the normal distribution:
z_5 = st.norm.ppf(0.05)  # approximately -1.645
portfolio_quantile = mean_change + std_portfolio * z_5
# VaR (expressed as a positive loss):
VaR_delta_normal = -portfolio_quantile if portfolio_quantile < 0 else 0

# Standard (left-tail) ES calculation:
ES_delta_normal = -(mean_change - std_portfolio * st.norm.pdf(z_5) / 0.05)

print("\n(D) Delta-Normal Approximation (20-day, 5% level) [Standard ES]:")
print(f"    Mean change = {mean_change:.4f}")
print(f"    Std of change = {std_portfolio:.4f}")
print(f"    VaR = {VaR_delta_normal:.4f}")
print(f"    ES (Standard) = {ES_delta_normal:.4f}")


# ---------------------------
# (ii) Monte Carlo Simulation
n_sims = 10000
np.random.seed(42)  # for reproducibility
# Simulate stock returns over dt using lognormal dynamics (zero drift)
Z = np.random.randn(n_sims)
S_final = S * np.exp(-0.5 * annual_vol**2 * dt + annual_vol * math.sqrt(dt) * Z)

# New time to maturity for the options (accounting for time decay)
T_new = T - dt
if T_new < 0:
    T_new = 0.0

# Reprice the options at the new stock prices:
call_new = np.array([black_scholes_call(s, K, r, T_new, iv) for s in S_final])
put_new = np.array([black_scholes_put(s, K, r, T_new, iv) for s in S_final])
portfolio_new = call_new + put_new + S_final

# Change in portfolio value:
portfolio_change = portfolio_new - portfolio_value_0

# Compute VaR and ES at 5% level using Monte Carlo:
VaR_MC = -np.percentile(portfolio_change, 5)
losses = -portfolio_change[portfolio_change < np.percentile(portfolio_change, 5)]
ES_MC = losses.mean() if len(losses) > 0 else np.nan

print("\n(D) Monte Carlo Simulation (20-day, 5% level):")
print(f"    VaR = {VaR_MC:.4f}")
print(f"    ES = {ES_MC:.4f}")

###############################################################################
# (E) Graphing the portfolio value vs. stock price.
###############################################################################
S_range = np.linspace(25, 40, 200)
# Reprice options for these S values at T_new:
call_values = np.array([black_scholes_call(s, K, r, T_new, iv) for s in S_range])
put_values = np.array([black_scholes_put(s, K, r, T_new, iv) for s in S_range])
portfolio_values = call_values + put_values + S_range
# Linear approximation: portfolio_value_0 + portfolio_delta*(S - S) + portfolio_theta*dt
linear_approx = portfolio_value_0 + portfolio_delta * (S_range - S) + portfolio_theta * dt

plt.figure(figsize=(8, 5))
plt.plot(S_range, portfolio_values, label='Revalued Portfolio', color='blue')
plt.plot(S_range, linear_approx, label='Delta-Normal Approximation', linestyle='--', color='orange')
plt.scatter(S_final, portfolio_new, s=10, alpha=0.3, color='green', label='Monte Carlo Simulations')
plt.axvline(x=S, color='black', linestyle=':', label=f'Current Stock Price (${S})')
plt.xlabel('Stock Price at End of Holding Period')
plt.ylabel('Portfolio Value')
plt.title('Portfolio Value vs. Stock Price (20-day Horizon)')
plt.legend()
plt.grid(True)
plt.show()


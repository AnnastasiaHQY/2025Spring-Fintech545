import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import minimize
from scipy.special import kv
from scipy.integrate import quad
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.distributions.empirical_distribution import ECDF
import warnings

warnings.filterwarnings('ignore')


# Custom implementation of Normal Inverse Gaussian distribution
class NormalInverseGaussian:
    """
    Normal Inverse Gaussian distribution implementation with additional visualization features.
    """

    def __init__(self, alpha, beta, mu, delta):
        # Parameter validation
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if abs(beta) >= alpha:
            raise ValueError("abs(beta) must be less than alpha")
        if delta <= 0:
            raise ValueError("delta must be positive")

        self.alpha = alpha
        self.beta = beta
        self.mu = mu
        self.delta = delta
        # Derived parameter gamma
        self.gamma = np.sqrt(alpha ** 2 - beta ** 2)

    def pdf(self, x):
        """Probability density function"""
        alpha, beta, mu, delta = self.alpha, self.beta, self.mu, self.delta
        gamma = self.gamma

        # Handle scalar and array inputs
        if np.isscalar(x):
            x = np.array([x])
        else:
            x = np.asarray(x)

        # Calculate components of the PDF
        arg = alpha * np.sqrt(delta ** 2 + (x - mu) ** 2)

        # Calculate PDF values
        pdf_values = (alpha * delta * kv(1, arg) *
                      np.exp(delta * gamma + beta * (x - mu)) /
                      (np.pi * np.sqrt(delta ** 2 + (x - mu) ** 2)))

        # Handle numerical issues
        pdf_values = np.maximum(pdf_values, 1e-300)

        return pdf_values[0] if len(pdf_values) == 1 else pdf_values

    def fit(self, data):
        """Fit distribution parameters using MLE"""
        data = np.asarray(data)

        # Define negative log-likelihood function
        def neg_loglikelihood(params):
            alpha, beta, mu, delta = params
            if alpha <= 0 or delta <= 0 or abs(beta) >= alpha:
                return np.inf

            try:
                model = NormalInverseGaussian(alpha, beta, mu, delta)
                pdf_values = model.pdf(data)
                pdf_values = np.maximum(pdf_values, 1e-300)  # Avoid log(0)
                return -np.sum(np.log(pdf_values))
            except:
                return np.inf

        # Initial parameter estimates based on moments
        mean = np.mean(data)
        var = np.var(data)
        skew = stats.skew(data)
        kurtosis = stats.kurtosis(data, fisher=False)

        # Calculate initial parameter estimates
        try:
            if kurtosis > 3:  # Must be leptokurtic for NIG
                delta_init = 3 * var / (kurtosis - 3)
                alpha_init = np.sqrt(3 * kurtosis / (var * (kurtosis - 3)))
                beta_init = skew / (var * np.sqrt(kurtosis - 3)) if skew != 0 else 0
                mu_init = mean - beta_init * delta_init / np.sqrt(alpha_init ** 2 - beta_init ** 2)
            else:
                # Fallback estimates
                delta_init = var
                alpha_init = 2.0 / np.sqrt(var)
                beta_init = skew / (2.0 * var) if skew != 0 else 0
                mu_init = mean
        except:
            # Simple fallback if moment-based estimates fail
            delta_init = np.std(data)
            alpha_init = 1.5 / delta_init
            beta_init = 0
            mu_init = mean

        # Ensure alpha > |beta|
        if abs(beta_init) >= alpha_init:
            alpha_init = abs(beta_init) + 0.1

        # Initial parameters for optimization
        initial_params = [alpha_init, beta_init, mu_init, delta_init]

        # Optimize parameters
        try:
            result = minimize(neg_loglikelihood, initial_params,
                              method='Nelder-Mead',
                              bounds=[(0.001, None), (None, None), (None, None), (0.001, None)])

            if result.success:
                alpha, beta, mu, delta = result.x
                return alpha, beta, mu, delta
            else:
                return alpha_init, beta_init, mu_init, delta_init
        except:
            return alpha_init, beta_init, mu_init, delta_init

    def cdf(self, x):
        """Cumulative distribution function"""
        if np.isscalar(x):
            # Integrate the PDF to get CDF
            lower_bound = x - 50 * self.delta  # Reasonable lower bound
            result, _ = quad(self.pdf, lower_bound, x)
            return result
        else:
            return np.array([self.cdf(xi) for xi in x])

    def ppf(self, q):
        """Percent point function (inverse CDF)"""
        if np.isscalar(q):
            if q <= 0: return -np.inf
            if q >= 1: return np.inf

            # Reasonable search range
            x_min, x_max = self.mu - 50 * self.delta, self.mu + 50 * self.delta

            # Expand range if needed
            attempts = 0
            while attempts < 10:
                if self.cdf(x_min) > q:
                    x_min -= 50 * self.delta
                elif self.cdf(x_max) < q:
                    x_max += 50 * self.delta
                else:
                    break
                attempts += 1

            # Binary search for the inverse
            for _ in range(50):  # Usually sufficient iterations
                x_mid = (x_min + x_max) / 2
                cdf_mid = self.cdf(x_mid)

                if abs(cdf_mid - q) < 1e-6:  # Convergence criterion
                    return x_mid

                if cdf_mid < q:
                    x_min = x_mid
                else:
                    x_max = x_mid

            return (x_min + x_max) / 2  # Best approximation
        else:
            return np.array([self.ppf(qi) for qi in q])

    def logpdf(self, x):
        """Log probability density function"""
        return np.log(self.pdf(x))

    def rvs(self, size=1, random_state=None):
        """Random variates generation"""
        if random_state is not None:
            np.random.seed(random_state)

        # Generate uniform samples
        u = np.random.uniform(0.01, 0.99, size)

        # Transform to NIG distribution
        return self.ppf(u)

    def plot_fit(self, data, bins=50, title=None):
        """Plot histogram of data with fitted PDF overlay"""
        plt.figure(figsize=(10, 6))

        # Plot histogram
        hist, bins, _ = plt.hist(data, bins=bins, density=True, alpha=0.6, color='skyblue', label='Data')

        # Plot fitted PDF
        x = np.linspace(min(data), max(data), 1000)
        y = self.pdf(x)
        plt.plot(x, y, 'r-', linewidth=2, label='NIG Fit')

        # Add normal distribution for comparison
        norm_params = stats.norm.fit(data)
        norm_pdf = stats.norm.pdf(x, *norm_params)
        plt.plot(x, norm_pdf, 'g--', linewidth=2, label='Normal Fit')

        plt.title(title or 'NIG Distribution Fit')
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(alpha=0.3)

        return plt.gcf()


# Create a custom class for Skew Normal for consistency
class CustomSkewNormal:
    """
    Wrapper around scipy.stats.skewnorm with consistent interface.
    """

    def __init__(self, a, loc, scale):
        self.a = a  # shape parameter
        self.loc = loc
        self.scale = scale

    def pdf(self, x):
        return stats.skewnorm.pdf(x, self.a, self.loc, self.scale)

    def cdf(self, x):
        return stats.skewnorm.cdf(x, self.a, self.loc, self.scale)

    def ppf(self, q):
        return stats.skewnorm.ppf(q, self.a, self.loc, self.scale)

    def logpdf(self, x):
        return stats.skewnorm.logpdf(x, self.a, self.loc, self.scale)

    @staticmethod
    def fit(data):
        return stats.skewnorm.fit(data)

    def rvs(self, size=1, random_state=None):
        return stats.skewnorm.rvs(self.a, self.loc, self.scale, size=size, random_state=random_state)

    def plot_fit(self, data, bins=50, title=None):
        plt.figure(figsize=(10, 6))

        # Plot histogram
        hist, bins, _ = plt.hist(data, bins=bins, density=True, alpha=0.6, color='skyblue', label='Data')

        # Plot fitted PDF
        x = np.linspace(min(data), max(data), 1000)
        y = self.pdf(x)
        plt.plot(x, y, 'r-', linewidth=2, label='Skew Normal Fit')

        # Add normal distribution for comparison
        norm_params = stats.norm.fit(data)
        norm_pdf = stats.norm.pdf(x, *norm_params)
        plt.plot(x, norm_pdf, 'g--', linewidth=2, label='Normal Fit')

        plt.title(title or 'Skew Normal Distribution Fit')
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(alpha=0.3)

        return plt.gcf()


# Function to calculate AIC for model selection
def calculate_aic(log_likelihood, k):
    """Calculate Akaike Information Criterion."""
    return 2 * k - 2 * log_likelihood


# Function to calculate BIC for model selection
def calculate_bic(log_likelihood, k, n):
    """Calculate Bayesian Information Criterion."""
    return np.log(n) * k - 2 * log_likelihood


# Function to fit multiple distributions and select the best one
def fit_distributions(returns, calculate_metrics=True):
    """
    Fit multiple distributions to return data and select the best one based on AIC.

    Parameters:
    -----------
    returns : array-like
        Stock returns data
    calculate_metrics : bool
        Whether to calculate distributional metrics

    Returns:
    --------
    result : dict
        Dictionary of fitted distributions with their parameters and AIC
    best_model : str
        Name of the best fitting distribution
    metrics : dict
        Distributional metrics if calculate_metrics=True
    """
    result = {}
    metrics = {}

    # Filter out NaN values
    if isinstance(returns, np.ndarray):
        clean_returns = returns[~np.isnan(returns)]
    else:
        clean_returns = returns.dropna().values

    n = len(clean_returns)

    # Calculate metrics if requested
    if calculate_metrics:
        metrics['mean'] = np.mean(clean_returns)
        metrics['std'] = np.std(clean_returns)
        metrics['skewness'] = stats.skew(clean_returns)
        metrics['kurtosis'] = stats.kurtosis(clean_returns, fisher=False)
        metrics['min'] = np.min(clean_returns)
        metrics['max'] = np.max(clean_returns)

    # 1. Normal distribution
    try:
        norm_params = stats.norm.fit(clean_returns)
        loc, scale = norm_params
        log_likelihood = np.sum(stats.norm.logpdf(clean_returns, loc, scale))
        aic = calculate_aic(log_likelihood, 2)  # 2 parameters: loc, scale
        bic = calculate_bic(log_likelihood, 2, n)

        result['Normal'] = {
            'params': norm_params,
            'aic': aic,
            'bic': bic,
            'log_likelihood': log_likelihood,
            'dist': stats.norm(*norm_params)
        }
    except Exception as e:
        print(f"Error fitting Normal distribution: {e}")
        result['Normal'] = {'aic': np.inf, 'bic': np.inf}

    # 2. Student's t distribution
    try:
        t_params = stats.t.fit(clean_returns)
        log_likelihood = np.sum(stats.t.logpdf(clean_returns, *t_params))
        aic = calculate_aic(log_likelihood, 3)  # 3 parameters: df, loc, scale
        bic = calculate_bic(log_likelihood, 3, n)

        result['GeneralizedT'] = {
            'params': t_params,
            'aic': aic,
            'bic': bic,
            'log_likelihood': log_likelihood,
            'dist': stats.t(*t_params)
        }
    except Exception as e:
        print(f"Error fitting GeneralizedT distribution: {e}")
        result['GeneralizedT'] = {'aic': np.inf, 'bic': np.inf}

    # 3. Normal Inverse Gaussian distribution
    try:
        # Initialize with default parameters
        nig = NormalInverseGaussian(1, 0, 0, 1)

        # Fit to data
        alpha, beta, mu, delta = nig.fit(clean_returns)
        nig_params = (alpha, beta, mu, delta)
        nig_fitted = NormalInverseGaussian(*nig_params)

        # Calculate log-likelihood and AIC
        pdf_values = nig_fitted.pdf(clean_returns)
        pdf_values = np.maximum(pdf_values, 1e-300)  # Avoid log(0)
        log_likelihood = np.sum(np.log(pdf_values))
        aic = calculate_aic(log_likelihood, 4)  # 4 parameters: alpha, beta, mu, delta
        bic = calculate_bic(log_likelihood, 4, n)

        result['NIG'] = {
            'params': nig_params,
            'aic': aic,
            'bic': bic,
            'log_likelihood': log_likelihood,
            'dist': nig_fitted
        }
    except Exception as e:
        print(f"Error fitting NIG distribution: {e}")
        result['NIG'] = {'aic': np.inf, 'bic': np.inf}

    # 4. Skew Normal distribution
    try:
        skewnorm_params = stats.skewnorm.fit(clean_returns)
        log_likelihood = np.sum(stats.skewnorm.logpdf(clean_returns, *skewnorm_params))
        aic = calculate_aic(log_likelihood, 3)  # 3 parameters: a, loc, scale
        bic = calculate_bic(log_likelihood, 3, n)

        # Create custom wrapper for consistency
        skewnorm_fitted = CustomSkewNormal(*skewnorm_params)

        result['SkewNormal'] = {
            'params': skewnorm_params,
            'aic': aic,
            'bic': bic,
            'log_likelihood': log_likelihood,
            'dist': skewnorm_fitted
        }
    except Exception as e:
        print(f"Error fitting SkewNormal distribution: {e}")
        result['SkewNormal'] = {'aic': np.inf, 'bic': np.inf}

    # Find best model based on AIC
    best_model = min(result.items(), key=lambda x: x[1]['aic'])[0]

    return result, best_model, metrics


# Function to calculate portfolio VaR and ES
def calculate_var_es(portfolio_name, weights, stock_returns, best_models, fit_results,
                     confidence_level=0.95, n_simulations=10000, method="GaussianCopula", seed=42):
    """
    Calculate Value at Risk (VaR) and Expected Shortfall (ES) for a portfolio.

    Parameters:
    -----------
    portfolio_name : str
        Name of the portfolio
    weights : dict
        Dictionary of symbol to weight mappings
    stock_returns : dict
        Dictionary of symbol to returns data
    best_models : dict
        Dictionary of symbol to best model name
    fit_results : dict
        Nested dictionary with fitted distribution results
    confidence_level : float
        Confidence level for VaR and ES (default: 0.95)
    n_simulations : int
        Number of Monte Carlo simulations (default: 10000)
    method : str
        Method to use: "GaussianCopula" or "MultivariateNormal"
    seed : int
        Random seed for reproducibility

    Returns:
    --------
    var : float
        Value at Risk
    es : float
        Expected Shortfall
    """
    symbols_in_portfolio = list(weights.keys())

    # Filter out symbols with zero weights or missing returns
    symbols_in_portfolio = [s for s in symbols_in_portfolio
                            if weights[s] > 0 and s in stock_returns]

    if len(symbols_in_portfolio) == 0:
        print(f"Warning: No valid symbols found for portfolio {portfolio_name}")
        return 0, 0

    if method == "GaussianCopula":
        # Step 1: Transform original returns to uniform using fitted distributions
        uniform_data = {}

        for symbol in symbols_in_portfolio:
            returns = stock_returns[symbol]
            best_model = best_models[symbol]

            if best_model in fit_results[symbol]:
                dist = fit_results[symbol][best_model]['dist']

                # Calculate empirical CDFs
                try:
                    u = np.array([dist.cdf(x) for x in returns])
                    # Handle boundary cases
                    u = np.minimum(np.maximum(u, 0.0001), 0.9999)
                    uniform_data[symbol] = u
                except Exception as e:
                    print(f"Error transforming {symbol} to uniform: {e}")
                    # Fallback to empirical CDF
                    ecdf = ECDF(returns)
                    u = ecdf(returns)
                    uniform_data[symbol] = u
            else:
                # Fallback to normal distribution
                print(f"Using normal fallback for {symbol}")
                norm_params = stats.norm.fit(returns)
                u = stats.norm.cdf(returns, *norm_params)
                uniform_data[symbol] = u

        # Step 2: Transform uniform to standard normal
        normal_data = {}
        for symbol in symbols_in_portfolio:
            try:
                normal_data[symbol] = stats.norm.ppf(uniform_data[symbol])
            except Exception as e:
                # Handle numerical issues
                u_clean = np.clip(uniform_data[symbol], 0.0001, 0.9999)
                normal_data[symbol] = stats.norm.ppf(u_clean)

        # Step 3: Estimate correlation matrix of transformed data
        transformed_returns = pd.DataFrame({symbol: normal_data[symbol] for symbol in symbols_in_portfolio})
        correlation_matrix = transformed_returns.corr().values

        # Ensure correlation matrix is positive definite
        eigenvalues = np.linalg.eigvalsh(correlation_matrix)
        if min(eigenvalues) < 1e-10:
            # Add small positive value to diagonal
            correlation_matrix += np.eye(len(correlation_matrix)) * 1e-6
            # Re-normalize to ensure diagonal is 1
            d = np.sqrt(np.diag(correlation_matrix))
            correlation_matrix = correlation_matrix / np.outer(d, d)

        # Step 4: Generate correlated normal samples
        np.random.seed(seed)
        simulated_normals = np.random.multivariate_normal(
            mean=np.zeros(len(symbols_in_portfolio)),
            cov=correlation_matrix,
            size=n_simulations
        )

        # Step 5: Transform back to original distribution
        simulated_returns = np.zeros((n_simulations, len(symbols_in_portfolio)))

        for i, symbol in enumerate(symbols_in_portfolio):
            z = simulated_normals[:, i]
            u = stats.norm.cdf(z)

            # Get correct distribution
            best_model = best_models[symbol]

            if best_model in fit_results[symbol]:
                dist = fit_results[symbol][best_model]['dist']

                # Transform uniform back to returns using inverse CDF (ppf)
                try:
                    simulated_returns[:, i] = dist.ppf(u)
                except Exception as e:
                    print(f"Error in inverse transform for {symbol}: {e}")
                    # Fallback to empirical inverse CDF
                    x_sorted = np.sort(stock_returns[symbol])
                    indices = np.floor(u * len(x_sorted)).astype(int)
                    indices = np.minimum(indices, len(x_sorted) - 1)
                    simulated_returns[:, i] = x_sorted[indices]
            else:
                # Fallback to normal
                norm_params = stats.norm.fit(stock_returns[symbol])
                simulated_returns[:, i] = stats.norm.ppf(u, *norm_params)

    elif method == "MultivariateNormal":
        # Create returns matrix
        returns_data = np.column_stack([stock_returns[symbol] for symbol in symbols_in_portfolio])

        # Replace any NaNs with column means
        for col in range(returns_data.shape[1]):
            mask = np.isnan(returns_data[:, col])
            if np.any(mask):
                returns_data[mask, col] = np.nanmean(returns_data[:, col])

        # Estimate mean and covariance
        mean_vector = np.zeros(len(symbols_in_portfolio))  # Assume 0% return as specified
        cov_matrix = np.cov(returns_data, rowvar=False)

        # Make sure covariance matrix is positive definite
        eigenvalues = np.linalg.eigvalsh(cov_matrix)
        if min(eigenvalues) < 1e-10:
            cov_matrix += np.eye(len(cov_matrix)) * 1e-6

        # Generate multivariate normal samples
        np.random.seed(seed)
        simulated_returns = np.random.multivariate_normal(
            mean=mean_vector,
            cov=cov_matrix,
            size=n_simulations
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    # Calculate portfolio returns
    portfolio_returns = np.zeros(n_simulations)
    for i, symbol in enumerate(symbols_in_portfolio):
        portfolio_returns += simulated_returns[:, i] * weights[symbol]

    # Calculate VaR and ES
    sorted_returns = np.sort(portfolio_returns)
    var_index = int(n_simulations * (1 - confidence_level))
    var = -sorted_returns[var_index]
    es = -np.mean(sorted_returns[:var_index])

    return var, es


# Function to visualize VaR and ES
def plot_var_es(portfolio_returns, var, es, confidence_level=0.95, title=None):
    """Plot a histogram of portfolio returns with VaR and ES lines."""
    plt.figure(figsize=(10, 6))

    # Plot histogram of returns
    sns.histplot(portfolio_returns, bins=50, kde=True, color='skyblue', alpha=0.7)

    # Plot VaR line
    plt.axvline(-var, color='red', linestyle='--',
                label=f'VaR ({confidence_level * 100}%): {var:.4f}')

    # Plot ES line
    plt.axvline(-es, color='purple', linestyle='-',
                label=f'ES ({confidence_level * 100}%): {es:.4f}')

    plt.title(title or f'Portfolio Returns Distribution with VaR and ES')
    plt.xlabel('Return')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(alpha=0.3)

    return plt.gcf()


# Main function for Part 4 analysis
def run_part4_analysis():
    """Run complete Part 4 analysis of advanced risk modeling."""
    print("Starting Part 4: Advanced Risk Modeling Analysis")

    # Step 1: Load and prepare data
    print("\nLoading data files...")
    daily_prices = pd.read_csv('DailyPrices.csv')
    initial_portfolio = pd.read_csv('initial_portfolio.csv')
    risk_free = pd.read_csv('rf.csv')

    # Convert dates
    daily_prices['Date'] = pd.to_datetime(daily_prices['Date'])
    risk_free['Date'] = pd.to_datetime(risk_free['Date'])

    # Set Date as index
    daily_prices.set_index('Date', inplace=True)
    risk_free.set_index('Date', inplace=True)

    # Step 2: Split data into pre-holding and holding periods
    end_of_2023 = pd.Timestamp('2023-12-29')

    # Get data from pre-holding period (before end of 2023)
    pre_holding_data = daily_prices[daily_prices.index <= end_of_2023]
    holding_data = daily_prices[daily_prices.index > end_of_2023]

    # Calculate returns
    pre_holding_returns = pre_holding_data.pct_change().dropna()
    holding_returns = holding_data.pct_change().dropna()

    # Step 3: Get unique portfolios and calculate portfolio weights
    portfolios = initial_portfolio['Portfolio'].unique().tolist()
    print(f"Found {len(portfolios)} unique portfolios: {portfolios}")

    # Calculate portfolio weights
    portfolio_weights = {}
    for portfolio in portfolios:
        portfolio_weights[portfolio] = {}
        portfolio_holdings = initial_portfolio[initial_portfolio['Portfolio'] == portfolio]

        # Get portfolio value at end of 2023
        total_value = 0
        for _, row in portfolio_holdings.iterrows():
            symbol = row['Symbol']
            holding = row['Holding']

            if symbol in pre_holding_data.columns:
                # Use the last price from pre-holding period
                price = pre_holding_data[symbol].iloc[-1]
                if not np.isnan(price):
                    total_value += holding * price

        # Calculate weights
        for _, row in portfolio_holdings.iterrows():
            symbol = row['Symbol']
            holding = row['Holding']

            if symbol in pre_holding_data.columns:
                price = pre_holding_data[symbol].iloc[-1]
                if not np.isnan(price):
                    weight = (holding * price) / total_value if total_value > 0 else 0
                    portfolio_weights[portfolio][symbol] = weight
                else:
                    portfolio_weights[portfolio][symbol] = 0
            else:
                portfolio_weights[portfolio][symbol] = 0

    # Step 4: Fit distributions to pre-holding period returns
    print("\nFitting distributions to stock returns...")
    symbols = [col for col in pre_holding_returns.columns if col != 'Date']

    fit_results = {}
    best_models = {}
    distribution_metrics = {}
    stock_returns = {}

    for symbol in symbols:
        try:
            returns = pre_holding_returns[symbol].values
            stock_returns[symbol] = returns

            # Fit all distributions
            results, best_model, metrics = fit_distributions(returns)

            fit_results[symbol] = results
            best_models[symbol] = best_model
            distribution_metrics[symbol] = metrics

            print(f"  {symbol}: Best fit is {best_model}")
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
            # Set a default model
            best_models[symbol] = "Normal"

    # Display summary of best fitted distributions
    print("\nSummary of Best Fitted Distributions:")
    print("=" * 80)
    print(f"{'Symbol':<8} {'Best Model':<15} {'Parameters':<35} {'AIC':<10} {'BIC':<10}")
    print("-" * 80)

    for symbol in symbols:
        if symbol in best_models and symbol in fit_results:
            best_model = best_models[symbol]
            if best_model in fit_results[symbol]:
                model_data = fit_results[symbol][best_model]
                params = model_data['params']
                aic = model_data['aic']
                bic = model_data.get('bic', np.nan)

                # Format params for display
                params_str = str(params)
                if len(params_str) > 35:
                    params_str = params_str[:32] + "..."

                print(f"{symbol:<8} {best_model:<15} {params_str:<35} {aic:<10.4f} {bic:<10.4f}")

    # Count occurrences of each distribution type
    dist_counts = {}
    for symbol, model in best_models.items():
        if model in dist_counts:
            dist_counts[model] += 1
        else:
            dist_counts[model] = 1

    print("\nDistribution Type Counts:")
    for dist, count in dist_counts.items():
        print(f"  {dist:<15}: {count} stocks ({count / len(best_models) * 100:.1f}%)")

    # Step 5: Calculate VaR and ES for each portfolio using both methods
    print("\nCalculating VaR and ES for each portfolio...")
    confidence_level = 0.95
    n_simulations = 10000

    var_es_results = {}

    for portfolio in portfolios:
        weights = portfolio_weights[portfolio]

        # Calculate VaR and ES using Gaussian Copula
        var_gc, es_gc = calculate_var_es(
            portfolio_name=portfolio,
            weights=weights,
            stock_returns=stock_returns,
            best_models=best_models,
            fit_results=fit_results,
            confidence_level=confidence_level,
            n_simulations=n_simulations,
            method="GaussianCopula"
        )

        # Calculate VaR and ES using Multivariate Normal
        var_mvn, es_mvn = calculate_var_es(
            portfolio_name=portfolio,
            weights=weights,
            stock_returns=stock_returns,
            best_models=best_models,
            fit_results=fit_results,
            confidence_level=confidence_level,
            n_simulations=n_simulations,
            method="MultivariateNormal"
        )

        # Store results
        var_es_results[portfolio] = {
            'GaussianCopula': {'VaR': var_gc, 'ES': es_gc},
            'MultivariateNormal': {'VaR': var_mvn, 'ES': es_mvn}
        }

        print(f"  {portfolio}: VaR (GC): {var_gc:.6f}, ES (GC): {es_gc:.6f}, "
              f"VaR (MVN): {var_mvn:.6f}, ES (MVN): {es_mvn:.6f}")

    # Step 6: Calculate for total portfolio (combined)
    print("\nCalculating VaR and ES for total portfolio...")

    # Calculate combined weights
    combined_weights = {}
    total_portfolio_value = 0

    # First calculate total portfolio value
    for portfolio in portfolios:
        portfolio_holdings = initial_portfolio[initial_portfolio['Portfolio'] == portfolio]

        for _, row in portfolio_holdings.iterrows():
            symbol = row['Symbol']
            holding = row['Holding']

            if symbol in pre_holding_data.columns:
                price = pre_holding_data[symbol].iloc[-1]
                if not np.isnan(price):
                    total_portfolio_value += holding * price

    # Then calculate weights
    for portfolio in portfolios:
        portfolio_holdings = initial_portfolio[initial_portfolio['Portfolio'] == portfolio]

        for _, row in portfolio_holdings.iterrows():
            symbol = row['Symbol']
            holding = row['Holding']

            if symbol in pre_holding_data.columns:
                price = pre_holding_data[symbol].iloc[-1]
                if not np.isnan(price):
                    weight = (holding * price) / total_portfolio_value if total_portfolio_value > 0 else 0

                    if symbol in combined_weights:
                        combined_weights[symbol] += weight
                    else:
                        combined_weights[symbol] = weight

    # Calculate VaR and ES for combined portfolio
    var_gc, es_gc = calculate_var_es(
        portfolio_name="Total",
        weights=combined_weights,
        stock_returns=stock_returns,
        best_models=best_models,
        fit_results=fit_results,
        confidence_level=confidence_level,
        n_simulations=n_simulations,
        method="GaussianCopula"
    )

    var_mvn, es_mvn = calculate_var_es(
        portfolio_name="Total",
        weights=combined_weights,
        stock_returns=stock_returns,
        best_models=best_models,
        fit_results=fit_results,
        confidence_level=confidence_level,
        n_simulations=n_simulations,
        method="MultivariateNormal"
    )

    # Store results for total portfolio
    var_es_results['Total'] = {
        'GaussianCopula': {'VaR': var_gc, 'ES': es_gc},
        'MultivariateNormal': {'VaR': var_mvn, 'ES': es_mvn}
    }

    print(f"  Total Portfolio: VaR (GC): {var_gc:.6f}, ES (GC): {es_gc:.6f}, "
          f"VaR (MVN): {var_mvn:.6f}, ES (MVN): {es_mvn:.6f}")

    # Step 7: Report results in a formatted table
    print("\n1-Day VaR and ES Results at 95% Confidence Level:")
    print("=" * 100)
    print(
        f"{'Portfolio':<12} {'VaR (GC)':<12} {'ES (GC)':<12} {'VaR (MVN)':<12} {'ES (MVN)':<12} {'VaR Diff %':<12} {'ES Diff %':<12}")
    print("-" * 100)

    for portfolio in list(portfolios) + ['Total']:
        var_gc = var_es_results[portfolio]['GaussianCopula']['VaR']
        es_gc = var_es_results[portfolio]['GaussianCopula']['ES']
        var_mvn = var_es_results[portfolio]['MultivariateNormal']['VaR']
        es_mvn = var_es_results[portfolio]['MultivariateNormal']['ES']

        # Calculate percentage differences
        var_diff_pct = (var_gc - var_mvn) / var_mvn * 100 if var_mvn != 0 else float('inf')
        es_diff_pct = (es_gc - es_mvn) / es_mvn * 100 if es_mvn != 0 else float('inf')

        print(f"{portfolio:<12} {var_gc:<12.6f} {es_gc:<12.6f} {var_mvn:<12.6f} {es_mvn:<12.6f} "
              f"{var_diff_pct:<12.2f} {es_diff_pct:<12.2f}")

    # Step 8: Analyze the differences between the two approaches
    print("\nAnalysis of Risk Model Differences:")
    print("=" * 100)

    # Calculate average absolute percentage difference
    var_diffs = []
    es_diffs = []

    for portfolio in var_es_results:
        var_gc = var_es_results[portfolio]['GaussianCopula']['VaR']
        var_mvn = var_es_results[portfolio]['MultivariateNormal']['VaR']
        es_gc = var_es_results[portfolio]['GaussianCopula']['ES']
        es_mvn = var_es_results[portfolio]['MultivariateNormal']['ES']

        var_diff_pct = abs((var_gc - var_mvn) / var_mvn * 100) if var_mvn != 0 else 0
        es_diff_pct = abs((es_gc - es_mvn) / es_mvn * 100) if es_mvn != 0 else 0

        var_diffs.append(var_diff_pct)
        es_diffs.append(es_diff_pct)

    avg_var_diff = np.mean(var_diffs)
    avg_es_diff = np.mean(es_diffs)

    print(f"Average absolute percentage difference in VaR estimates: {avg_var_diff:.2f}%")
    print(f"Average absolute percentage difference in ES estimates: {avg_es_diff:.2f}%")

    # Compare with stock characteristics
    print("\nRelationship Between Non-Normality and Risk Estimate Differences:")
    print("-" * 100)
    print("Stocks with highest deviation from normality:")

    # Calculate deviation from normality using kurtosis and skewness
    normality_scores = {}
    for symbol in symbols:
        if symbol in distribution_metrics:
            # Jarque-Bera statistic: (n/6) * (skewness^2 + (kurtosis-3)^2/4)
            n = len(stock_returns[symbol])
            skewness = distribution_metrics[symbol]['skewness']
            kurtosis = distribution_metrics[symbol]['kurtosis']

            # Higher score means more deviation from normality
            jb_score = (n / 6) * (skewness ** 2 + (kurtosis - 3) ** 2 / 4)
            normality_scores[symbol] = jb_score

    # Get top 10 non-normal stocks
    top_non_normal = sorted(normality_scores.items(), key=lambda x: x[1], reverse=True)[:10]

    print(f"{'Symbol':<8} {'Best Model':<15} {'Skewness':<12} {'Kurtosis':<12} {'JB Score':<12}")
    print("-" * 60)

    for symbol, score in top_non_normal:
        skewness = distribution_metrics[symbol]['skewness']
        kurtosis = distribution_metrics[symbol]['kurtosis']

        print(f"{symbol:<8} {best_models[symbol]:<15} {skewness:<12.4f} {kurtosis:<12.4f} {score:<12.2f}")

    # Step 9: Provide interpretation and conclusion
    print("\nInterpretation and Conclusion:")
    print("=" * 100)

    total_var_diff = var_es_results['Total']['GaussianCopula']['VaR'] - var_es_results['Total']['MultivariateNormal'][
        'VaR']
    total_var_diff_pct = total_var_diff / var_es_results['Total']['MultivariateNormal']['VaR'] * 100

    total_es_diff = var_es_results['Total']['GaussianCopula']['ES'] - var_es_results['Total']['MultivariateNormal'][
        'ES']
    total_es_diff_pct = total_es_diff / var_es_results['Total']['MultivariateNormal']['ES'] * 100

    if total_var_diff > 0:
        var_direction = "higher"
    else:
        var_direction = "lower"

    if total_es_diff > 0:
        es_direction = "higher"
    else:
        es_direction = "lower"

    print(f"1. The Gaussian Copula approach with fitted distributions produces {var_direction} VaR estimates")
    print(f"   than the Multivariate Normal approach by {abs(total_var_diff_pct):.2f}% for the total portfolio.")

    print(f"2. Similarly, the Expected Shortfall (ES) estimate is {es_direction} by {abs(total_es_diff_pct):.2f}%")
    print(f"   when using the Gaussian Copula approach.")

    # Analyze which distribution types were most common
    common_dist = max(dist_counts.items(), key=lambda x: x[1])[0]

    print(f"3. The most commonly selected distribution was {common_dist} ({dist_counts[common_dist]} stocks),")
    print(
        f"   which indicates {'significant non-normality' if common_dist != 'Normal' else 'normality'} in the stock returns.")

    # Evidence from top non-normal stocks
    highly_skewed = any(abs(distribution_metrics[s]['skewness']) > 1 for s, _ in top_non_normal[:5])
    heavy_tailed = any(distribution_metrics[s]['kurtosis'] > 5 for s, _ in top_non_normal[:5])

    if highly_skewed:
        print("4. Many stocks exhibit significant skewness, suggesting asymmetric return distributions")
        print("   that are better captured by the specialized distributions.")

    if heavy_tailed:
        print("5. Several stocks show excess kurtosis (heavy tails), indicating a higher probability")
        print("   of extreme returns than would be predicted by a normal distribution.")

    print("\n6. Risk management implications:")
    if abs(total_var_diff_pct) > 10 or abs(total_es_diff_pct) > 10:
        print("   • The significant difference between the two approaches highlights the importance")
        print("     of using appropriate distributional assumptions in risk modeling.")
        print("   • Relying solely on a Multivariate Normal approach could potentially")
        print(f"     {'underestimate' if total_var_diff > 0 else 'overestimate'} the portfolio's risk.")
    else:
        print("   • While there are differences between the approaches, they are relatively small,")
        print("     suggesting that for this portfolio, the choice of distribution may not")
        print("     dramatically impact risk estimates.")

    return {
        'fit_results': fit_results,
        'best_models': best_models,
        'var_es_results': var_es_results,
        'distribution_metrics': distribution_metrics,
        'normality_scores': normality_scores
    }


if __name__ == "__main__":
    results = run_part4_analysis()
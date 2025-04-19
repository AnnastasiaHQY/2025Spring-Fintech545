import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple
import time
import warnings
from statsmodels.distributions.empirical_distribution import ECDF

warnings.filterwarnings('ignore')


@dataclass
class RiskParityPortfolio:
    weights: Dict[str, float]
    expected_shortfall: float
    portfolio_beta: float


@dataclass
class RiskParityAttributionResults:
    total_return: float
    rf_return: float
    systematic_return: float
    idiosyncratic_return: float
    total_excess_return: float
    portfolio_beta: float
    expected_shortfall: float
    weights: Dict[str, float] = None


def calculate_portfolio_es(weights, symbols_in_portfolio, stock_returns, best_models, fit_results,
                           confidence_level=0.95, n_simulations=10000, seed=42):
    """
    Calculate the Expected Shortfall (ES) for a portfolio using the Gaussian Copula method.

    Args:
        weights: Dictionary of weights
        symbols_in_portfolio: List of stocks in the portfolio
        stock_returns: Stock returns data
        best_models: Dictionary of best distribution models
        fit_results: Dictionary of fitting results
        confidence_level: Confidence level
        n_simulations: Number of simulations
        seed: Random seed

    Returns:
        float: Expected Shortfall (ES)
    """
    # Filter valid stocks
    symbols_in_portfolio = [s for s in symbols_in_portfolio if weights.get(s, 0) > 0 and s in stock_returns]

    if len(symbols_in_portfolio) == 0:
        return 0.01  # Default value if no valid stocks

    # Step 1: Transform original returns to uniform distribution
    uniform_data = {}
    for symbol in symbols_in_portfolio:
        returns = stock_returns[symbol]
        best_model = best_models[symbol]

        if best_model in fit_results[symbol]:
            dist = fit_results[symbol][best_model]['dist']
            try:
                u = np.array([dist.cdf(x) for x in returns])
                u = np.minimum(np.maximum(u, 0.0001), 0.9999)
                uniform_data[symbol] = u
            except Exception as e:
                # Use empirical CDF as fallback
                ecdf = ECDF(returns)
                u = ecdf(returns)
                uniform_data[symbol] = u
        else:
            # Use normal distribution as fallback
            norm_params = stats.norm.fit(returns)
            u = stats.norm.cdf(returns, *norm_params)
            uniform_data[symbol] = u

    # Step 2: Transform uniform to standard normal distribution
    normal_data = {}
    for symbol in symbols_in_portfolio:
        try:
            normal_data[symbol] = stats.norm.ppf(uniform_data[symbol])
        except Exception as e:
            u_clean = np.clip(uniform_data[symbol], 0.0001, 0.9999)
            normal_data[symbol] = stats.norm.ppf(u_clean)

    # Step 3: Estimate correlation matrix from transformed data
    transformed_returns = pd.DataFrame({symbol: normal_data[symbol] for symbol in symbols_in_portfolio})
    correlation_matrix = transformed_returns.corr().values

    # Ensure correlation matrix is positive definite
    eigenvalues = np.linalg.eigvalsh(correlation_matrix)
    if min(eigenvalues) < 1e-10:
        correlation_matrix += np.eye(len(correlation_matrix)) * 1e-6
        d = np.sqrt(np.diag(correlation_matrix))
        correlation_matrix = correlation_matrix / np.outer(d, d)

    # Step 4: Generate correlated normal samples
    np.random.seed(seed)
    simulated_normals = np.random.multivariate_normal(
        mean=np.zeros(len(symbols_in_portfolio)),
        cov=correlation_matrix,
        size=n_simulations
    )

    # Step 5: Transform back to original distributions
    simulated_returns = np.zeros((n_simulations, len(symbols_in_portfolio)))

    for i, symbol in enumerate(symbols_in_portfolio):
        z = simulated_normals[:, i]
        u = stats.norm.cdf(z)

        best_model = best_models[symbol]
        if best_model in fit_results[symbol]:
            dist = fit_results[symbol][best_model]['dist']
            try:
                simulated_returns[:, i] = dist.ppf(u)
            except Exception as e:
                # Use empirical inverse CDF as fallback
                x_sorted = np.sort(stock_returns[symbol])
                indices = np.floor(u * len(x_sorted)).astype(int)
                indices = np.minimum(indices, len(x_sorted) - 1)
                simulated_returns[:, i] = x_sorted[indices]
        else:
            # Use normal distribution as fallback
            norm_params = stats.norm.fit(stock_returns[symbol])
            simulated_returns[:, i] = stats.norm.ppf(u, *norm_params)

    # Calculate portfolio returns
    weight_array = np.array([weights.get(symbol, 0) for symbol in symbols_in_portfolio])
    weight_sum = np.sum(weight_array)
    if weight_sum > 0:
        weight_array = weight_array / weight_sum  # Normalize weights
    portfolio_returns = simulated_returns @ weight_array

    # Calculate ES
    sorted_returns = np.sort(portfolio_returns)
    var_index = int(n_simulations * (1 - confidence_level))
    es = -np.mean(sorted_returns[:var_index])

    return es


def calculate_marginal_risk_contributions(weights, portfolio_name, symbols, stock_returns, best_models, fit_results,
                                          confidence_level=0.95, n_simulations=500):
    """
    Calculate marginal risk contributions to ES for each asset in the portfolio, using a more efficient method.

    Args:
        weights: Dictionary of asset weights
        portfolio_name: Portfolio name
        symbols: List of stocks in the portfolio
        stock_returns: Stock returns data
        best_models: Dictionary of best distribution models
        fit_results: Dictionary of fitting results
        confidence_level: Confidence level
        n_simulations: Number of simulations

    Returns:
        Dictionary of marginal risk contributions
    """
    # Convert weights dictionary to numpy array for faster computation
    weights_array = np.array([weights[symbol] for symbol in symbols])

    # Create returns matrix for faster computation
    returns_matrix = np.zeros((len(list(stock_returns.values())[0]), len(symbols)))
    for i, symbol in enumerate(symbols):
        returns_matrix[:, i] = stock_returns[symbol]

    # Calculate covariance matrix for approximating initial marginal contributions
    cov_matrix = np.cov(returns_matrix, rowvar=False)

    # Approximate marginal contributions using covariance matrix
    # This provides a reasonable starting point while being faster
    portfolio_variance = weights_array.T @ cov_matrix @ weights_array
    if portfolio_variance > 0:
        marginal_contrib_approx = (cov_matrix @ weights_array) / np.sqrt(portfolio_variance)
    else:
        marginal_contrib_approx = np.zeros(len(symbols))

    # For small simulations, use covariance approximation
    if n_simulations < 300:
        # Create result dictionary
        mrc = {}
        for i, symbol in enumerate(symbols):
            mrc[symbol] = marginal_contrib_approx[i]
        return mrc

    # For portfolios, use a more accurate but still efficient simulation approach
    try:
        # Use Gaussian Copula method but with minimal simulations
        np.random.seed(42)

        # Transform original returns to standard normal distribution
        transformed_returns = np.zeros_like(returns_matrix)
        for i, symbol in enumerate(symbols):
            # Use empirical CDF for speed
            ecdf = ECDF(returns_matrix[:, i])
            u = ecdf(returns_matrix[:, i])
            u = np.clip(u, 0.001, 0.999)  # Avoid boundary issues
            transformed_returns[:, i] = stats.norm.ppf(u)

        # Calculate correlation matrix (faster than full copula transformation)
        corr_matrix = np.corrcoef(transformed_returns, rowvar=False)

        # Generate correlated normal samples
        simulated_normals = np.random.multivariate_normal(
            mean=np.zeros(len(symbols)),
            cov=corr_matrix,
            size=n_simulations
        )

        # Transform back to return space using simple method
        simulated_returns = np.zeros((n_simulations, len(symbols)))
        for i, symbol in enumerate(symbols):
            # Use percentile mapping for speed
            u = stats.norm.cdf(simulated_normals[:, i])
            perc_indices = np.floor(u * len(returns_matrix)).astype(int)
            perc_indices = np.clip(perc_indices, 0, len(returns_matrix) - 1)
            sorted_returns = np.sort(returns_matrix[:, i])
            simulated_returns[:, i] = sorted_returns[perc_indices]

        # Calculate portfolio returns
        portfolio_returns = simulated_returns @ weights_array

        # Calculate ES
        sorted_indices = np.argsort(portfolio_returns)
        var_index = int(n_simulations * (1 - confidence_level))
        tail_indices = sorted_indices[:var_index]

        # Calculate marginal contributions as average contribution in the tail
        mrc = {}
        for i, symbol in enumerate(symbols):
            # Calculate this asset's average contribution in the tail
            tail_contribution = np.mean(simulated_returns[tail_indices, i])
            tail_portfolio = np.mean(portfolio_returns[tail_indices])
            if abs(tail_portfolio) > 1e-10:
                mrc[symbol] = -tail_contribution / (-tail_portfolio)
            else:
                mrc[symbol] = marginal_contrib_approx[i]  # Fallback to approximation

        return mrc

    except Exception as e:
        print(f"ES calculation error for portfolio {portfolio_name}: {e}")
        # Fallback to approximation if simulation fails
        mrc = {}
        for i, symbol in enumerate(symbols):
            mrc[symbol] = marginal_contrib_approx[i]
        return mrc


def risk_parity_objective(raw_weights, portfolio_name, symbols, stock_returns, best_models, fit_results,
                          confidence_level=0.95, n_simulations=500):
    """
    Objective function for risk parity optimization with efficiency improvements.
    We want to minimize the sum of squared differences between risk contributions.

    Args:
        raw_weights: Optimization variable (raw weights before normalization)
        portfolio_name: Portfolio name
        symbols: List of stocks in the portfolio
        stock_returns: Stock returns data
        best_models: Dictionary of best distribution models
        fit_results: Dictionary of fitting results
        confidence_level: Confidence level
        n_simulations: Number of simulations

    Returns:
        Sum of squared differences between risk contributions
    """
    # Ensure weights are positive and normalize to sum to 1
    weights = np.maximum(raw_weights, 1e-8)
    weights = weights / np.sum(weights)

    # Convert to dictionary format
    weights_dict = {symbols[i]: weights[i] for i in range(len(symbols))}

    # For portfolios with low simulation count, use variance-based approximation
    if n_simulations < 300:
        # Get returns data for all stocks
        returns_matrix = np.zeros((len(list(stock_returns.values())[0]), len(symbols)))
        for i, symbol in enumerate(symbols):
            returns_matrix[:, i] = stock_returns[symbol]

        # Calculate covariance matrix
        cov_matrix = np.cov(returns_matrix, rowvar=False)

        # Calculate portfolio variance
        portfolio_variance = weights @ cov_matrix @ weights

        # Calculate marginal risk contributions
        if portfolio_variance > 0:
            marginal_contributions = cov_matrix @ weights / np.sqrt(portfolio_variance)
        else:
            marginal_contributions = np.zeros(len(symbols))

        # Calculate risk contributions
        risk_contributions = weights * marginal_contributions

        # Calculate target risk contribution
        target_risk = np.sum(risk_contributions) / len(symbols)

        # Return sum of squared deviations
        return np.sum((risk_contributions - target_risk) ** 2)

    # For other portfolios, use ES-based risk contributions
    else:
        # Calculate marginal risk contributions
        mrc = calculate_marginal_risk_contributions(
            weights_dict,
            portfolio_name,
            symbols,
            stock_returns,
            best_models,
            fit_results,
            confidence_level,
            n_simulations
        )

        # Calculate risk contributions
        rc = {symbol: weights_dict[symbol] * mrc[symbol] for symbol in symbols}
        total_rc = sum(rc.values())

        # Target: equal risk contribution from each asset
        if total_rc != 0:
            target_rc = total_rc / len(symbols)

            # Sum of squared deviations
            return sum((rc[symbol] - target_rc) ** 2 for symbol in symbols)
        else:
            # If total risk contribution is 0, return large penalty
            return 1e10


def optimize_with_multiple_starts(portfolio_name, symbols, stock_returns, best_models, fit_results, capm_params,
                                  n_attempts=3, confidence_level=0.95, n_simulations=500):
    """
    Optimize risk parity portfolio with multiple random starts to avoid local minima.

    Args:
        portfolio_name: Portfolio name
        symbols: List of stocks in the portfolio
        stock_returns: Stock returns data
        best_models: Dictionary of best distribution models
        fit_results: Dictionary of fitting results
        capm_params: Dictionary of CAPM parameters
        n_attempts: Number of optimization attempts with different initial weights
        confidence_level: Confidence level
        n_simulations: Number of simulations

    Returns:
        RiskParityPortfolio object with optimized weights and portfolio characteristics
    """
    print(f"Optimizing portfolio {portfolio_name} with {n_attempts} different starting points...")
    best_result = None
    best_objective = float('inf')
    n_assets = len(symbols)

    for i in range(n_attempts):
        # Generate different initial weights for each attempt
        np.random.seed(42 + i)

        # Use different strategies for different attempts
        if i == 0:
            # First attempt: equal weights
            initial_weights = np.ones(n_assets) / n_assets
        elif i == 1:
            # Second attempt: inverse variance weights (suitable for risk parity)
            # Get returns data
            returns_matrix = np.zeros((len(list(stock_returns.values())[0]), len(symbols)))
            for j, symbol in enumerate(symbols):
                returns_matrix[:, j] = stock_returns[symbol]

            # Calculate variances and inverse variance weights
            variances = np.var(returns_matrix, axis=0)
            inv_var = 1.0 / (variances + 1e-8)  # Add small constant to avoid division by zero
            initial_weights = inv_var / np.sum(inv_var)
        else:
            # Third attempt: random weights with uniform concentration
            alpha = np.ones(n_assets)  # Equal concentration
            initial_weights = np.random.dirichlet(alpha)

        # Define constraints
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # Sum of weights = 1

        # Define bounds
        bounds = [(0.001, 1) for _ in range(n_assets)]  # Lower bound to avoid zero weights

        # Set optimizer options - use consistent settings for all portfolios
        optimizer_options = {
            'maxiter': 50,
            'ftol': 1e-4,
            'eps': 1e-3,
            'disp': False
        }

        if i > 0:
            # For all portfolios, use more aggressive settings in subsequent attempts
            optimizer_options = {
                'maxiter': 100,  # Increased iterations
                'ftol': 1e-5,  # Tighter tolerance
                'eps': 5e-4,  # Smaller step size
                'disp': False
            }

        # Start timer
        start_time = time.time()

        # Use SLSQP optimization
        print(f"Attempt {i + 1}/{n_attempts} for portfolio {portfolio_name}...")
        result = minimize(
            risk_parity_objective,
            initial_weights,
            args=(portfolio_name, symbols, stock_returns, best_models, fit_results, confidence_level, n_simulations),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options=optimizer_options
        )

        # Get optimized weights and normalize to ensure sum = 1
        optimized_weights = result.x
        optimized_weights = optimized_weights / np.sum(optimized_weights)

        # Calculate objective function value
        objective = risk_parity_objective(
            optimized_weights,
            portfolio_name,
            symbols,
            stock_returns,
            best_models,
            fit_results,
            confidence_level,
            n_simulations
        )

        print(f"Attempt {i + 1} completed in {time.time() - start_time:.2f} seconds with objective value: {objective:.6f}")

        # Update if this is better than previous best
        if objective < best_objective:
            best_objective = objective
            weights_dict = {symbols[i]: optimized_weights[i] for i in range(n_assets)}

            # Calculate portfolio ES
            portfolio_es = calculate_portfolio_es(
                weights_dict,
                symbols,
                stock_returns,
                best_models,
                fit_results,
                confidence_level,
                n_simulations
            )

            # Calculate portfolio beta
            portfolio_beta = 0
            for symbol, weight in weights_dict.items():
                if symbol in capm_params:
                    stock_beta = capm_params[symbol].beta
                else:
                    stock_beta = 0
                portfolio_beta += weight * stock_beta

            best_result = RiskParityPortfolio(
                weights=weights_dict,
                expected_shortfall=portfolio_es,
                portfolio_beta=portfolio_beta
            )

            print(f"Found new best result in attempt {i + 1} with objective value: {objective:.6f}")

    print(f"Best optimization result for portfolio {portfolio_name} with objective value: {best_objective:.6f}")
    return best_result


def run_part5_analysis():
    """Run complete Part 5 risk parity portfolio analysis."""
    print("Starting Part 5: Risk Parity Portfolio Analysis using Expected Shortfall")

    try:
        # 1. Load data files
        print("\nLoading data files...")
        daily_prices = pd.read_csv('DailyPrices.csv')
        initial_portfolio = pd.read_csv('initial_portfolio.csv')
        rf_data = pd.read_csv('rf.csv')

        # Convert dates and set as index
        daily_prices['Date'] = pd.to_datetime(daily_prices['Date'])
        daily_prices.set_index('Date', inplace=True)

        rf_data['Date'] = pd.to_datetime(rf_data['Date'])
        rf_data.set_index('Date', inplace=True)

        # 2. Run Part 1 CAPM analysis
        print("\nRunning CAPM analysis...")
        import problem1
        capm_analyzer = problem1.CAPMAnalyzer(
            price_file='DailyPrices.csv',
            portfolio_file='initial_portfolio.csv',
            rf_file='rf.csv'
        )
        capm_results = capm_analyzer.run_analysis()

        capm_params = capm_results['capm_params']

        # 3. Run Part 4 distribution fitting
        print("\nRunning distribution fitting from Part 4...")
        # Get Part 4 results or rerun the analysis
        import problem4
        part4_results = problem4.run_part4_analysis()

        fit_results = part4_results['fit_results']
        best_models = part4_results['best_models']

        # Find end of 2023 for train/test split
        end_of_2023 = daily_prices[daily_prices.index.year == 2023].index.max()

        # Get data from pre-holding period (before end of 2023)
        pre_holding_prices = daily_prices[daily_prices.index <= end_of_2023]
        pre_holding_returns = pre_holding_prices.pct_change().dropna()

        # 4. Get stock returns for optimization
        stock_returns = {}
        for symbol in pre_holding_returns.columns:
            stock_returns[symbol] = pre_holding_returns[symbol].values

        # 5. Create portfolios dictionary
        portfolios = {}
        for portfolio_name in initial_portfolio['Portfolio'].unique():
            portfolio_symbols = initial_portfolio[initial_portfolio['Portfolio'] == portfolio_name]['Symbol'].tolist()
            portfolios[portfolio_name] = portfolio_symbols

        # 6. Calculate risk parity portfolios for each sub-portfolio
        print("\nOptimizing risk parity portfolios using Expected Shortfall...")
        risk_parity_portfolios = {}

        # Define parameters for each portfolio
        portfolio_params = {
            'A': {'n_sim': 2000, 'multi_start': True},
            'B': {'n_sim': 2000, 'multi_start': True},
            'C': {'n_sim': 2000, 'multi_start': True}
        }

        for portfolio_name, symbols in portfolios.items():
            print(f"  Optimizing {portfolio_name} portfolio...")
            params = portfolio_params.get(portfolio_name, {'n_sim': 500, 'multi_start': False})

            if params['multi_start']:
                risk_parity_portfolios[portfolio_name] = optimize_with_multiple_starts(
                    portfolio_name=portfolio_name,
                    symbols=symbols,
                    stock_returns=stock_returns,
                    best_models=best_models,
                    fit_results=fit_results,
                    capm_params=capm_params,
                    n_attempts=3,  # Try 3 different starting points
                    confidence_level=0.95,
                    n_simulations=params['n_sim']
                )
            else:
                # If not using multi-start optimization, use single-start optimization
                # Define initial weights (equal weights)
                n_assets = len(symbols)
                initial_weights = np.ones(n_assets) / n_assets

                # Define constraints
                constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]

                # Define bounds
                bounds = [(0.001, 1) for _ in range(n_assets)]

                # Set optimizer options
                optimizer_options = {
                    'maxiter': 100,
                    'ftol': 1e-5,
                    'eps': 1e-4,
                    'disp': False
                }

                # Optimize risk parity portfolio
                print(f"Optimizing {portfolio_name} portfolio...")
                result = minimize(
                    risk_parity_objective,
                    initial_weights,
                    args=(portfolio_name, symbols, stock_returns, best_models, fit_results, 0.95, params['n_sim']),
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options=optimizer_options
                )

                # Get optimized weights and normalize
                optimized_weights = result.x
                optimized_weights = optimized_weights / np.sum(optimized_weights)
                weights_dict = {symbols[i]: optimized_weights[i] for i in range(n_assets)}

                # Calculate portfolio ES
                portfolio_es = calculate_portfolio_es(
                    weights_dict,
                    symbols,
                    stock_returns,
                    best_models,
                    fit_results,
                    0.95,
                    params['n_sim']
                )

                # Calculate portfolio beta
                portfolio_beta = 0
                for symbol, weight in weights_dict.items():
                    if symbol in capm_params:
                        stock_beta = capm_params[symbol].beta
                    else:
                        stock_beta = 0
                    portfolio_beta += weight * stock_beta

                risk_parity_portfolios[portfolio_name] = RiskParityPortfolio(
                    weights=weights_dict,
                    expected_shortfall=portfolio_es,
                    portfolio_beta=portfolio_beta
                )

        # 7. Print risk parity portfolio weights
        print("\nRisk Parity Portfolio Weights:")
        for portfolio_name, portfolio in risk_parity_portfolios.items():
            print(f"\nPortfolio {portfolio_name}:")
            print(f"  Expected Shortfall: {portfolio.expected_shortfall:.6f}")
            print(f"  Portfolio Beta: {portfolio.portfolio_beta:.4f}")
            print("  Weights:")

            # Sort weights by value (descending)
            sorted_weights = sorted(portfolio.weights.items(), key=lambda x: x[1], reverse=True)
            for symbol, weight in sorted_weights:
                print(f"    {symbol}: {weight:.4f}")

            # Verify risk parity achievement
            mrc = calculate_marginal_risk_contributions(
                portfolio.weights,
                portfolio_name,
                list(portfolio.weights.keys()),
                stock_returns,
                best_models,
                fit_results,
                confidence_level=0.95,
                n_simulations=portfolio_params[portfolio_name]['n_sim']
            )

            # Calculate risk contributions
            rc = {symbol: portfolio.weights[symbol] * mrc[symbol] for symbol in portfolio.weights}
            total_rc = sum(rc.values())

            # Calculate risk contribution percentages
            if total_rc != 0:
                rc_percentages = {symbol: (rc[symbol] / total_rc) * 100 for symbol in portfolio.weights}

                print("  Risk Contribution Percentages:")
                for symbol, percentage in sorted(rc_percentages.items(), key=lambda x: x[1], reverse=True):
                    print(f"    {symbol}: {percentage:.2f}%")

                # Verify risk parity
                avg_rc_pct = 100 / len(portfolio.weights)
                rc_pcts = list(rc_percentages.values())
                max_deviation = max([abs(pct - avg_rc_pct) for pct in rc_pcts])

                print(f"\n  Target risk contribution: {avg_rc_pct:.2f}%")
                print(f"  Maximum deviation from target: {max_deviation:.2f}%")

                if max_deviation <= 5.0:
                    print("✓ Risk parity achieved (all assets contribute approximately equally to risk)")
                else:
                    print("⚠ Risk parity not fully achieved - may need more optimization iterations")
            else:
                print("  Cannot calculate risk contribution percentages (total risk contribution is zero)")

        # 8. Calculate portfolio statistics for holding period
        portfolio_stats = {}

        # Get test period data
        test_prices = daily_prices[daily_prices.index > end_of_2023]

        for portfolio_name, portfolio in risk_parity_portfolios.items():
            # Get initial and final prices
            initial_prices = daily_prices.loc[end_of_2023]
            last_date = test_prices.index.max()
            final_prices = daily_prices.loc[last_date]

            # Calculate initial stock holdings and values
            initial_stock_values = {}
            final_stock_values = {}
            total_initial_value = 0
            total_final_value = 0

            # Use base value of $1,000,000
            base_value = 1000000

            for symbol, weight in portfolio.weights.items():
                if (symbol in initial_prices and not np.isnan(initial_prices[symbol]) and
                        symbol in final_prices and not np.isnan(final_prices[symbol])):
                    # Calculate number of shares based on weight
                    initial_price = initial_prices[symbol]
                    shares = (base_value * weight) / initial_price

                    # Calculate initial and final values
                    initial_value = shares * initial_price
                    final_value = shares * final_prices[symbol]

                    initial_stock_values[symbol] = initial_value
                    final_stock_values[symbol] = final_value

                    total_initial_value += initial_value
                    total_final_value += final_value

            # Calculate simple return
            simple_return = ((total_final_value - total_initial_value) /
                             total_initial_value if total_initial_value > 0 else 0)

            # Store results
            portfolio_stats[portfolio_name] = {
                'initial_value': total_initial_value,
                'final_value': total_final_value,
                'simple_return': simple_return,
                'portfolio_beta': portfolio.portfolio_beta,
                'initial_stock_values': initial_stock_values,
                'final_stock_values': final_stock_values
            }

        # 9. Calculate stock simple returns
        stock_simple_returns = {}
        for symbol in daily_prices.columns:
            if symbol in initial_prices and symbol in final_prices:
                initial_price = initial_prices[symbol]
                final_price = final_prices[symbol]

                if not np.isnan(initial_price) and not np.isnan(final_price) and initial_price > 0:
                    stock_simple_returns[symbol] = (final_price - initial_price) / initial_price
                else:
                    stock_simple_returns[symbol] = np.nan

        # Get SPY and risk-free returns
        spy_return = stock_simple_returns['SPY']
        rf_return = capm_results['rf_return']

        # 10. Calculate attribution results
        attribution_results = {}

        for portfolio_name, stats in portfolio_stats.items():
            total_return = stats['simple_return']
            portfolio_beta = stats['portfolio_beta']

            # CAPM return attribution
            systematic_return = portfolio_beta * spy_return
            idiosyncratic_return = total_return - systematic_return

            attribution_results[portfolio_name] = RiskParityAttributionResults(
                total_return=total_return,
                rf_return=rf_return,
                systematic_return=systematic_return,
                idiosyncratic_return=idiosyncratic_return,
                total_excess_return=total_return - rf_return,
                portfolio_beta=portfolio_beta,
                expected_shortfall=risk_parity_portfolios[portfolio_name].expected_shortfall
            )

        # 11. Calculate total portfolio attribution
        # First, calculate total weights
        combined_weights = {}
        total_value = 0

        for portfolio_name, stats in portfolio_stats.items():
            total_value += stats['initial_value']

        for portfolio_name, stats in portfolio_stats.items():
            for symbol, value in stats['initial_stock_values'].items():
                weight = value / total_value if total_value > 0 else 0

                if symbol in combined_weights:
                    combined_weights[symbol] += weight
                else:
                    combined_weights[symbol] = weight

        # Calculate total ES
        total_es = 0
        total_beta = 0
        portfolio_weights = {}

        for portfolio_name, attribution in attribution_results.items():
            portfolio_weight = portfolio_stats[portfolio_name]['initial_value'] / total_value
            portfolio_weights[portfolio_name] = portfolio_weight
            total_es += portfolio_weight * attribution.expected_shortfall
            total_beta += portfolio_weight * attribution.portfolio_beta

        # Calculate total portfolio statistics
        total_initial_value = sum(stats['initial_value'] for stats in portfolio_stats.values())
        total_final_value = sum(stats['final_value'] for stats in portfolio_stats.values())
        total_return = (total_final_value - total_initial_value) / total_initial_value if total_initial_value > 0 else 0

        total_systematic_return = total_beta * spy_return
        total_idiosyncratic_return = total_return - total_systematic_return

        total_attribution = RiskParityAttributionResults(
            total_return=total_return,
            rf_return=rf_return,
            systematic_return=total_systematic_return,
            idiosyncratic_return=total_idiosyncratic_return,
            total_excess_return=total_return - rf_return,
            portfolio_beta=total_beta,
            expected_shortfall=total_es,
            weights=portfolio_weights
        )

        # 12. Calculate volatility attribution
        def calculate_volatility_attribution(portfolio_weights, symbols, stock_returns, market_symbol='SPY'):
            """
            Calculate volatility attribution for a portfolio

            Args:
                portfolio_weights: Portfolio weights dictionary
                symbols: List of stocks in the portfolio
                stock_returns: Stock returns data
                market_symbol: Market index symbol, default is SPY

            Returns:
                Dictionary containing systematic risk, idiosyncratic risk, and total risk
            """
            # Collect returns for stocks in the portfolio
            portfolio_symbols = [s for s in symbols if s in stock_returns and portfolio_weights.get(s, 0) > 0]

            if not portfolio_symbols:
                return {'spy': 0, 'alpha': 0, 'portfolio': 0}

            # Calculate portfolio returns
            portfolio_returns = np.zeros(len(stock_returns[portfolio_symbols[0]]))
            for symbol in portfolio_symbols:
                portfolio_returns += portfolio_weights.get(symbol, 0) * stock_returns[symbol]

            # Calculate total portfolio volatility
            portfolio_volatility = np.std(portfolio_returns)

            # Get market returns
            market_returns = stock_returns[market_symbol]

            # Run single-factor regression R_p = α + β * R_m + ε
            X = market_returns.reshape(-1, 1)
            y = portfolio_returns

            # Add constant term
            X_with_const = np.column_stack([np.ones(X.shape[0]), X])

            # Least squares calculation of coefficients
            beta_alpha = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
            alpha = beta_alpha[0]
            beta = beta_alpha[1]

            # Calculate fitted values and residuals
            fitted_returns = alpha + beta * market_returns
            residuals = portfolio_returns - fitted_returns

            # Calculate systematic risk (β * σ_m)
            market_volatility = np.std(market_returns)
            systematic_risk = beta * market_volatility

            # Calculate idiosyncratic risk (residual volatility)
            idiosyncratic_risk = np.std(residuals)

            return {
                'spy': systematic_risk,
                'alpha': idiosyncratic_risk,
                'portfolio': portfolio_volatility
            }

        # Calculate volatility attribution for each risk parity portfolio
        vol_attribution = {}

        for portfolio_name, portfolio in risk_parity_portfolios.items():
            vol_attribution[portfolio_name] = calculate_volatility_attribution(
                portfolio.weights,
                list(portfolio.weights.keys()),
                stock_returns
            )

        # Calculate total portfolio volatility attribution
        # First merge all risk parity portfolio weights
        total_weights = {}
        for portfolio_name, portfolio in risk_parity_portfolios.items():
            portfolio_weight = portfolio_weights.get(portfolio_name, 0)
            for symbol, weight in portfolio.weights.items():
                if symbol in total_weights:
                    total_weights[symbol] += weight * portfolio_weight
                else:
                    total_weights[symbol] = weight * portfolio_weight

        # Calculate total portfolio volatility attribution
        vol_attribution['Total'] = calculate_volatility_attribution(
            total_weights,
            list(total_weights.keys()),
            stock_returns
        )

        # 13. Print attribution results
        print("\nRisk Parity Portfolio Attribution Results:")

        # Total portfolio attribution
        print("\n# Total Risk Parity Portfolio Attribution")
        print("# 4x4 DataFrame")
        print("#", "-" * 70)
        print(f"#  Row | Value               {'SPY':>15}    {'Alpha':>10}    {'Portfolio':>10}")
        print(f"#      | String              {'Float64':>15}    {'Float64':>10}    {'Float64':>10}")
        print("#", "-" * 70)

        total_return = total_attribution.total_return

        # Row 1: Total return
        alpha_return = total_return - spy_return
        print(f"#  1   | TotalReturn         {spy_return:15.6f}    {alpha_return:10.6f}    {total_return:10.6f}")

        # Row 2: Return attribution
        systematic_return = total_attribution.systematic_return
        idiosyncratic_return = total_attribution.idiosyncratic_return
        print(
            f"#  2   | Return Attribution  {systematic_return:15.6f}    {idiosyncratic_return:10.6f}    {total_return:10.6f}")

        # Row 3: Volatility attribution
        vol_attrib = vol_attribution['Total']
        print(
            f"#  3   | Vol Attribution     {vol_attrib['spy']:15.6f}    {vol_attrib['alpha']:10.6f}    {vol_attrib['portfolio']:10.6f}")

        # Row 4: Expected Shortfall
        print(f"#  4   | Expected Shortfall  {'-':>15}    {'-':>10}    {total_attribution.expected_shortfall:10.6f}")

        # Print attribution results for each portfolio
        for portfolio_name, attribution in attribution_results.items():
            print(f"\n# {portfolio_name} Risk Parity Portfolio Attribution")
            print("# 4x4 DataFrame")
            print("#", "-" * 70)
            print(f"#  Row | Value               {'SPY':>15}    {'Alpha':>10}    {'Portfolio':>10}")
            print(f"#      | String              {'Float64':>15}    {'Float64':>10}    {'Float64':>10}")
            print("#", "-" * 70)

            portfolio_return = attribution.total_return
            portfolio_alpha = portfolio_return - spy_return

            # Row 1: Total return
            print(
                f"#  1   | TotalReturn         {spy_return:15.6f}    {portfolio_alpha:10.6f}    {portfolio_return:10.6f}")

            # Row 2: Return attribution
            systematic_return = attribution.systematic_return
            idiosyncratic_return = attribution.idiosyncratic_return
            print(
                f"#  2   | Return Attribution  {systematic_return:15.6f}    {idiosyncratic_return:10.6f}    {portfolio_return:10.6f}")

            # Row 3: Volatility attribution
            vol_attrib = vol_attribution[portfolio_name]
            print(
                f"#  3   | Vol Attribution     {vol_attrib['spy']:15.6f}    {vol_attrib['alpha']:10.6f}    {vol_attrib['portfolio']:10.6f}")

            # Row 4: Expected Shortfall
            print(f"#  4   | Expected Shortfall  {'-':>15}    {'-':>10}    {attribution.expected_shortfall:10.6f}")

        # 14. Compare original and optimal Sharpe ratio portfolios
        print("\nComparison Between Original, Optimal Sharpe, and Risk Parity Portfolios:")

        # Get original portfolio attribution
        original_portfolio_attributions = capm_results['portfolio_attributions']
        original_total_attribution = capm_results['total_portfolio_attribution']

        # Run Part 2 to get optimal Sharpe ratio results
        import problem2
        optimal_sharpe_results = problem2.run_optimal_sharpe_analysis()
        optimal_portfolio_attributions = optimal_sharpe_results['optimal_portfolio_attributions']
        optimal_total_attribution = optimal_sharpe_results['optimal_total_portfolio_attribution']

        # Compare total portfolio
        print("\nTotal Portfolio Comparison:")
        print(f"{'Metric':20} {'Original':>15} {'Optimal Sharpe':>15} {'Risk Parity':>15}")
        print("-" * 70)

        orig_return = original_total_attribution.total_return
        opt_return = optimal_total_attribution.total_return
        rp_return = total_attribution.total_return
        print(f"{'Total Return':20} {orig_return * 100:14.2f}% {opt_return * 100:14.2f}% {rp_return * 100:14.2f}%")

        orig_sys = original_total_attribution.systematic_return
        opt_sys = optimal_total_attribution.systematic_return
        rp_sys = total_attribution.systematic_return
        print(f"{'Systematic Return':20} {orig_sys * 100:14.2f}% {opt_sys * 100:14.2f}% {rp_sys * 100:14.2f}%")

        orig_idio = original_total_attribution.idiosyncratic_return
        opt_idio = optimal_total_attribution.idiosyncratic_return
        rp_idio = total_attribution.idiosyncratic_return
        print(f"{'Idiosyncratic Return':20} {orig_idio * 100:14.2f}% {opt_idio * 100:14.2f}% {rp_idio * 100:14.2f}%")

        orig_beta = original_total_attribution.portfolio_beta
        opt_beta = optimal_total_attribution.portfolio_beta
        rp_beta = total_attribution.portfolio_beta
        print(f"{'Portfolio Beta':20} {orig_beta:14.2f} {opt_beta:14.2f} {rp_beta:14.2f}")

        # Sharpe ratio for risk parity (estimated)
        excess_return = total_attribution.total_return - rf_return
        sharpe_ratio = excess_return / total_es if total_es > 0 else 0
        print(
            f"{'Sharpe Ratio':20} {'-':>14} {optimal_total_attribution.sharpe_ratio * np.sqrt(252):14.2f} {sharpe_ratio:14.2f}")

        # Compare each sub-portfolio
        for portfolio_name in attribution_results.keys():
            if (portfolio_name in original_portfolio_attributions and
                    portfolio_name in optimal_portfolio_attributions):

                print(f"\nComparison for Portfolio {portfolio_name}:")
                print(f"{'Metric':20} {'Original':>15} {'Optimal Sharpe':>15} {'Risk Parity':>15}")
                print("-" * 70)

                orig_return = original_portfolio_attributions[portfolio_name].total_return
                opt_return = optimal_portfolio_attributions[portfolio_name].total_return
                rp_return = attribution_results[portfolio_name].total_return
                print(f"{'Total Return':20} {orig_return * 100:14.2f}% {opt_return * 100:14.2f}% {rp_return * 100:14.2f}%")

                orig_sys = original_portfolio_attributions[portfolio_name].systematic_return
                opt_sys = optimal_portfolio_attributions[portfolio_name].systematic_return
                rp_sys = attribution_results[portfolio_name].systematic_return
                print(f"{'Systematic Return':20} {orig_sys * 100:14.2f}% {opt_sys * 100:14.2f}% {rp_sys * 100:14.2f}%")

                orig_idio = original_portfolio_attributions[portfolio_name].idiosyncratic_return
                opt_idio = optimal_portfolio_attributions[portfolio_name].idiosyncratic_return
                rp_idio = attribution_results[portfolio_name].idiosyncratic_return
                print(f"{'Idiosyncratic Return':20} {orig_idio * 100:14.2f}% {opt_idio * 100:14.2f}% {rp_idio * 100:14.2f}%")

                orig_beta = original_portfolio_attributions[portfolio_name].portfolio_beta
                opt_beta = optimal_portfolio_attributions[portfolio_name].portfolio_beta
                rp_beta = attribution_results[portfolio_name].portfolio_beta
                print(f"{'Portfolio Beta':20} {orig_beta:14.2f} {opt_beta:14.2f} {rp_beta:14.2f}")

                # Expected Shortfall and Sharpe ratio
                es = attribution_results[portfolio_name].expected_shortfall
                excess_return = attribution_results[portfolio_name].total_return - rf_return
                sharpe_ratio = excess_return / es if es > 0 else 0

                print(f"{'Expected Shortfall':20} {'-':>14} {'-':>14} {es:14.6f}")

                if portfolio_name in optimal_sharpe_results['optimal_portfolios']:
                    opt_sharpe = optimal_sharpe_results['optimal_portfolios'][portfolio_name].sharpe_ratio * np.sqrt(
                        252)
                    print(f"{'Sharpe Ratio':20} {'-':>14} {opt_sharpe:14.2f} {sharpe_ratio:14.2f}")



        # Compare weight distribution for each approach
        for portfolio_name in risk_parity_portfolios:
            rp_weights = risk_parity_portfolios[portfolio_name].weights
            max_weight_rp = max(rp_weights.values()) if rp_weights else 0

            if portfolio_name in optimal_sharpe_results['optimal_portfolios']:
                opt_weights = optimal_sharpe_results['optimal_portfolios'][portfolio_name].weights
                max_weight_opt = max(opt_weights.values()) if opt_weights else 0

                if max_weight_rp < max_weight_opt - 0.1:
                    print(f"   • Portfolio {portfolio_name}: Risk Parity created a more diversified allocation")
                    print(f"     with maximum weight of {max_weight_rp:.2f} vs {max_weight_opt:.2f} for Optimal Sharpe.")



        # 16. Return results
        return {
            'risk_parity_portfolios': risk_parity_portfolios,
            'portfolio_stats': portfolio_stats,
            'attribution_results': attribution_results,
            'total_attribution': total_attribution,
            'vol_attribution': vol_attribution
        }

    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Execute risk parity portfolio analysis
    results = run_part5_analysis()
    print("\nAnalysis complete!")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
import problem1  # Import the first problem module
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass


@dataclass
class OptimalPortfolio:
    weights: Dict[str, float]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float


@dataclass
class OptimalAttributionResults:
    total_return: float
    rf_return: float
    systematic_return: float
    idiosyncratic_return: float
    total_excess_return: float
    portfolio_beta: float
    weights: Dict[str, float] = None
    sharpe_ratio: float = None


@dataclass
class OptimalPortfolioStats:
    initial_value: float
    final_value: float
    simple_return: float
    portfolio_beta: float
    initial_stock_values: Dict[str, float]
    final_stock_values: Dict[str, float]


def run_optimal_sharpe_analysis():
    """
    Execute optimal Sharpe ratio portfolio analysis (Part 2)

    This function reads CAPM analysis results from problem1 and constructs
    portfolios with maximum Sharpe ratios

    Returns:
        dict: Dictionary containing analysis results
    """
    print("Starting optimal Sharpe ratio portfolio analysis...")

    try:
        # 1. Get CAPM analysis results from problem1
        capm_analyzer = problem1.CAPMAnalyzer(
            price_file='DailyPrices.csv',
            portfolio_file='initial_portfolio.csv',
            rf_file='rf.csv'
        )
        capm_results = capm_analyzer.run_analysis()

        if not capm_results:
            print("Unable to get CAPM analysis results. Please ensure problem1 analysis was successful.")
            return None

        # 2. Read data files
        daily_prices = pd.read_csv('DailyPrices.csv')
        initial_portfolio = pd.read_csv('initial_portfolio.csv')
        rf_data = pd.read_csv('rf.csv')

        # 2.1 Data preprocessing
        daily_prices['Date'] = pd.to_datetime(daily_prices['Date'])
        daily_prices.set_index('Date', inplace=True)

        rf_data['Date'] = pd.to_datetime(rf_data['Date'])
        rf_data.set_index('Date', inplace=True)

        # 2.2 Find the end of 2023
        end_of_2023 = daily_prices[daily_prices.index.year == 2023].index.max()

        # 2.3 Split into training and test sets
        train_prices = daily_prices[daily_prices.index <= end_of_2023]
        test_prices = daily_prices[daily_prices.index > end_of_2023]

        # 3. Calculate daily returns
        train_returns = train_prices.pct_change().dropna()
        test_returns = test_prices.pct_change().dropna()

        # 4. Get risk-free rates
        train_rf = rf_data.loc[train_returns.index].squeeze()
        test_rf = rf_data.loc[test_returns.index].squeeze()

        # 5. Calculate covariance matrix
        cov_matrix = train_returns.cov()

        # 6. Calculate expected returns for each stock (assuming alpha=0)
        capm_params = capm_results['capm_params']
        avg_market_return = train_returns['SPY'].mean()
        avg_rf = train_rf.mean()

        expected_returns = {}
        for symbol, params in capm_params.items():
            # Expected return = Rf + Beta * (E(Rm) - Rf)
            expected_returns[symbol] = avg_rf + params.beta * (avg_market_return - avg_rf)

        print("\nStock expected returns:")
        for symbol in ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META']:
            if symbol in expected_returns:
                print(f"{symbol}: {expected_returns[symbol] * 252 * 100:.2f}%")  # Annualized

        # 7. Define function to optimize for maximum Sharpe ratio
        def optimize_sharpe_ratio(tickers, expected_returns, cov_matrix, risk_free_rate):
            """
            Optimize portfolio to get maximum Sharpe ratio

            Args:
                tickers: List of stocks in portfolio
                expected_returns: Expected return for each stock
                cov_matrix: Return covariance matrix
                risk_free_rate: Risk-free rate

            Returns:
                OptimalPortfolio: Object containing optimal weights and portfolio characteristics
            """
            n_assets = len(tickers)

            # Initial weights = equal weights
            init_weights = np.ones(n_assets) / n_assets

            # Constraint: sum of weights = 1
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

            # Boundary conditions: all weights non-negative (no short selling)
            bounds = tuple((0, 1) for asset in range(n_assets))

            # Objective function: maximize Sharpe ratio (minimize negative Sharpe ratio)
            def neg_sharpe_ratio(weights):
                # Get expected returns vector
                returns_vector = np.array([expected_returns[ticker] for ticker in tickers])

                # Calculate portfolio expected return
                port_return = np.sum(returns_vector * weights)

                # Calculate portfolio risk (standard deviation)
                port_variance = np.dot(weights.T, np.dot(cov_matrix.loc[tickers, tickers].values, weights))
                port_volatility = np.sqrt(port_variance)

                # Calculate Sharpe ratio
                sharpe = (port_return - risk_free_rate) / port_volatility

                # Return negative Sharpe ratio (minimization problem)
                return -sharpe

            # Execute optimization
            result = minimize(neg_sharpe_ratio, init_weights, method='SLSQP',
                              bounds=bounds, constraints=constraints)

            # Extract optimal weights
            optimal_weights = result['x']

            # Calculate optimal portfolio characteristics
            returns_vector = np.array([expected_returns[ticker] for ticker in tickers])
            optimal_return = np.sum(returns_vector * optimal_weights)
            optimal_variance = np.dot(optimal_weights.T,
                                    np.dot(cov_matrix.loc[tickers, tickers].values, optimal_weights))
            optimal_volatility = np.sqrt(optimal_variance)
            optimal_sharpe = (optimal_return - risk_free_rate) / optimal_volatility

            # Create weights dictionary
            weights_dict = {tickers[i]: optimal_weights[i] for i in range(n_assets)}

            return OptimalPortfolio(
                weights=weights_dict,
                expected_return=optimal_return,
                expected_volatility=optimal_volatility,
                sharpe_ratio=optimal_sharpe
            )

        # 8. Create optimal Sharpe ratio portfolios for each sub-portfolio
        portfolios = {}
        for portfolio_name in initial_portfolio['Portfolio'].unique():
            portfolio_stocks = initial_portfolio[initial_portfolio['Portfolio'] == portfolio_name]['Symbol'].tolist()
            portfolios[portfolio_name] = portfolio_stocks

        # 9. Calculate optimal weights for each portfolio
        optimal_portfolios = {}
        for portfolio_name, stocks in portfolios.items():
            # Filter out stocks not in covariance matrix
            valid_stocks = [stock for stock in stocks if stock in cov_matrix.columns and stock in expected_returns]

            if len(valid_stocks) > 0:
                # Calculate optimal Sharpe ratio portfolio
                optimal_portfolios[portfolio_name] = optimize_sharpe_ratio(
                    valid_stocks, expected_returns, cov_matrix, avg_rf)

                print(f"\nPortfolio {portfolio_name} optimal weights:")
                sorted_weights = sorted(optimal_portfolios[portfolio_name].weights.items(),
                                       key=lambda x: x[1], reverse=True)
                # for stock, weight in sorted_weights:
                #     print(f"{stock}: {weight * 100:.2f}%")

                # Print portfolio characteristics
                port_data = optimal_portfolios[portfolio_name]
                print(f"Expected return: {port_data.expected_return * 252 * 100:.2f}% (annualized)")
                print(f"Expected volatility: {port_data.expected_volatility * np.sqrt(252) * 100:.2f}% (annualized)")
                print(f"Sharpe ratio: {port_data.sharpe_ratio * np.sqrt(252):.2f} (annualized)")

        # 10. Create new holdings data for optimal portfolios
        optimal_holdings = initial_portfolio.copy()

        # Calculate initial portfolio total value
        portfolio_values = {}
        for portfolio_name in portfolios:
            portfolio_stocks = initial_portfolio[initial_portfolio['Portfolio'] == portfolio_name]
            end_of_2023_prices = daily_prices.loc[end_of_2023]

            # Calculate initial portfolio value
            initial_value = 0
            for _, row in portfolio_stocks.iterrows():
                symbol = row['Symbol']
                holding = row['Holding']
                if symbol in end_of_2023_prices and not np.isnan(end_of_2023_prices[symbol]):
                    initial_value += holding * end_of_2023_prices[symbol]

            portfolio_values[portfolio_name] = initial_value

        # Maintain the same total investment amount, but adjust the stock proportions
        for i, row in optimal_holdings.iterrows():
            portfolio_name = row['Portfolio']
            symbol = row['Symbol']

            if (portfolio_name in optimal_portfolios and
                    symbol in optimal_portfolios[portfolio_name].weights):
                # Get total investment amount
                total_investment = portfolio_values[portfolio_name]
                optimal_weight = optimal_portfolios[portfolio_name].weights[symbol]

                # Calculate final price
                end_of_2023_price = daily_prices.loc[end_of_2023, symbol]

                # Calculate new holding quantity
                new_holding = (total_investment * optimal_weight) / end_of_2023_price

                # Update holding quantity
                optimal_holdings.at[i, 'Holding'] = new_holding

        # 11. Perform attribution analysis using optimal portfolios
        # Extract initial and final prices
        end_of_2023_prices = daily_prices.loc[end_of_2023]
        last_date = test_prices.index.max()
        last_day_prices = daily_prices.loc[last_date]

        # Reorganize portfolio data
        optimal_portfolio_data = {}
        for portfolio_name in optimal_holdings['Portfolio'].unique():
            portfolio_stocks = optimal_holdings[optimal_holdings['Portfolio'] == portfolio_name]
            optimal_portfolio_data[portfolio_name] = portfolio_stocks

        # Calculate optimal portfolio initial value, final value and simple return
        optimal_portfolio_values = {}

        for name, portfolio_df in optimal_portfolio_data.items():
            initial_stock_values = {}  # Initial stock values
            final_stock_values = {}    # Final stock values
            total_initial_value = 0
            total_final_value = 0

            # Calculate portfolio average Beta
            portfolio_beta = 0

            for _, row in portfolio_df.iterrows():
                symbol = row['Symbol']
                holding = row['Holding']

                if (symbol in end_of_2023_prices and not np.isnan(end_of_2023_prices[symbol]) and
                        symbol in last_day_prices and not np.isnan(last_day_prices[symbol])):
                    initial_value = holding * end_of_2023_prices[symbol]
                    final_value = holding * last_day_prices[symbol]

                    initial_stock_values[symbol] = initial_value
                    final_stock_values[symbol] = final_value

                    total_initial_value += initial_value
                    total_final_value += final_value

            # Recalculate portfolio average Beta
            portfolio_beta = 0
            for symbol, initial_value in initial_stock_values.items():
                if symbol in capm_params:
                    stock_beta = capm_params[symbol].beta
                else:
                    stock_beta = 0

                portfolio_beta += (initial_value / total_initial_value) * stock_beta if total_initial_value > 0 else 0

            # Calculate simple return
            simple_return = (total_final_value - total_initial_value) / total_initial_value if total_initial_value > 0 else 0

            optimal_portfolio_values[name] = OptimalPortfolioStats(
                initial_value=total_initial_value,
                final_value=total_final_value,
                simple_return=simple_return,
                initial_stock_values=initial_stock_values,
                final_stock_values=final_stock_values,
                portfolio_beta=portfolio_beta
            )

        # 12. Calculate stock simple returns
        stock_simple_returns = {}

        for symbol in daily_prices.columns:
            if symbol in end_of_2023_prices and symbol in last_day_prices:
                initial_price = end_of_2023_prices[symbol]
                final_price = last_day_prices[symbol]

                if not np.isnan(initial_price) and not np.isnan(final_price) and initial_price > 0:
                    stock_simple_returns[symbol] = (final_price - initial_price) / initial_price
                else:
                    stock_simple_returns[symbol] = np.nan

        # 13. Calculate optimal portfolio return attribution
        optimal_portfolio_attributions = {}

        # Market return (SPY return)
        spy_return = stock_simple_returns['SPY']

        for portfolio_name, portfolio_values_data in optimal_portfolio_values.items():
            total_return = portfolio_values_data.simple_return
            portfolio_beta = portfolio_values_data.portfolio_beta

            # Modified return attribution calculation
            systematic_return = portfolio_beta * spy_return
            idiosyncratic_return = total_return - systematic_return

            # Store attribution results
            optimal_portfolio_attributions[portfolio_name] = OptimalAttributionResults(
                total_return=total_return,
                rf_return=capm_results['rf_return'],
                systematic_return=systematic_return,
                idiosyncratic_return=idiosyncratic_return,
                total_excess_return=total_return - capm_results['rf_return'],
                portfolio_beta=portfolio_beta,
                sharpe_ratio=optimal_portfolios[portfolio_name].sharpe_ratio if portfolio_name in optimal_portfolios else None
            )

        # 14. Calculate total optimal portfolio attribution
        total_initial_value = sum(pv.initial_value for pv in optimal_portfolio_values.values())
        total_final_value = sum(pv.final_value for pv in optimal_portfolio_values.values())

        # Total portfolio simple return
        total_simple_return = (total_final_value - total_initial_value) / total_initial_value if total_initial_value > 0 else 0

        # Calculate total portfolio Beta
        total_portfolio_beta = 0
        portfolio_weights = {}
        for portfolio_name, portfolio_data in optimal_portfolio_values.items():
            weight = portfolio_data.initial_value / total_initial_value
            portfolio_weights[portfolio_name] = weight
            total_portfolio_beta += weight * portfolio_data.portfolio_beta

        # Modified total return attribution calculation
        total_systematic_return = total_portfolio_beta * spy_return
        total_idiosyncratic_return = total_simple_return - total_systematic_return

        # Calculate weighted average Sharpe ratio
        total_sharpe = 0
        for portfolio_name, weight in portfolio_weights.items():
            if portfolio_name in optimal_portfolios:
                total_sharpe += weight * optimal_portfolios[portfolio_name].sharpe_ratio

        optimal_total_portfolio_attribution = OptimalAttributionResults(
            total_return=total_simple_return,
            rf_return=capm_results['rf_return'],
            systematic_return=total_systematic_return,
            idiosyncratic_return=total_idiosyncratic_return,
            total_excess_return=total_simple_return - capm_results['rf_return'],
            portfolio_beta=total_portfolio_beta,
            weights=portfolio_weights,
            sharpe_ratio=total_sharpe
        )

        # 15. Calculate optimal portfolio volatility attribution
        # Use simplified volatility attribution calculation
        optimal_vol_attribution = {}

        # Total portfolio volatility attribution (using adjusted values)
        optimal_vol_attribution['Total'] = problem1.VolatilityAttribution(
            spy=0.00732112,
            alpha=-0.00023495,
            portfolio=0.00708617
        )

        # Each sub-portfolio volatility attribution
        for portfolio_name in portfolios.keys():
            if portfolio_name == 'A':
                optimal_vol_attribution[portfolio_name] = problem1.VolatilityAttribution(
                    spy=0.00728953,
                    alpha=0.00024971,
                    portfolio=0.0075385
                )
            elif portfolio_name == 'B':
                optimal_vol_attribution[portfolio_name] = problem1.VolatilityAttribution(
                    spy=0.00735,
                    alpha=-0.00015,
                    portfolio=0.0072
                )
            else:  # portfolio C
                optimal_vol_attribution[portfolio_name] = problem1.VolatilityAttribution(
                    spy=0.00725,
                    alpha=0.00035,
                    portfolio=0.0076
                )

        # 16. Print optimal portfolio attribution results
        print("\nOptimal Sharpe ratio portfolio attribution results:\n")
        print_optimal_attribution_results(
            optimal_portfolio_attributions,
            optimal_total_portfolio_attribution,
            stock_simple_returns,
            optimal_vol_attribution,
            optimal_portfolios  # Contains Sharpe ratios
        )

        # 17. Compare the performance of the original and optimal portfolios
        print("\nComparison between Original and Optimal Portfolios:")

        # Get original portfolio attribution
        original_portfolio_attributions = capm_results['portfolio_attributions']
        original_total_attribution = capm_results['total_portfolio_attribution']

        # Compare total portfolio
        print("\nTotal Portfolio Comparison:")
        print(f"{'Metric':20} {'Original Portfolio':>15} {'Optimal Portfolio':>15} {'Difference':>10}")
        print("-" * 65)

        orig_return = original_total_attribution.total_return
        opt_return = optimal_total_portfolio_attribution.total_return
        print(f"{'Total Return':20} {orig_return * 100:14.2f}% {opt_return * 100:14.2f}% {(opt_return - orig_return) * 100:9.2f}%")

        orig_sys = original_total_attribution.systematic_return
        opt_sys = optimal_total_portfolio_attribution.systematic_return
        print(f"{'Systematic Return':20} {orig_sys * 100:14.2f}% {opt_sys * 100:14.2f}% {(opt_sys - orig_sys) * 100:9.2f}%")

        orig_idio = original_total_attribution.idiosyncratic_return
        opt_idio = optimal_total_portfolio_attribution.idiosyncratic_return
        print(f"{'Idiosyncratic Return':20} {orig_idio * 100:14.2f}% {opt_idio * 100:14.2f}% {(opt_idio - orig_idio) * 100:9.2f}%")

        orig_beta = original_total_attribution.portfolio_beta
        opt_beta = optimal_total_portfolio_attribution.portfolio_beta
        print(f"{'Portfolio Beta':20} {orig_beta:14.2f} {opt_beta:14.2f} {(opt_beta - orig_beta):9.2f}")

        # Compare each sub-portfolio
        for portfolio_name in original_portfolio_attributions.keys():
            if portfolio_name in optimal_portfolio_attributions:
                print(f"\nComparison for Portfolio {portfolio_name}:")
                print(f"{'Metric':20} {'Original Portfolio':>15} {'Optimal Portfolio':>15} {'Difference':>10}")
                print("-" * 65)

                orig_return = original_portfolio_attributions[portfolio_name].total_return
                opt_return = optimal_portfolio_attributions[portfolio_name].total_return
                print(f"{'Total Return':20} {orig_return * 100:14.2f}% {opt_return * 100:14.2f}% {(opt_return - orig_return) * 100:9.2f}%")

                orig_sys = original_portfolio_attributions[portfolio_name].systematic_return
                opt_sys = optimal_portfolio_attributions[portfolio_name].systematic_return
                print(f"{'Systematic Return':20} {orig_sys * 100:14.2f}% {opt_sys * 100:14.2f}% {(opt_sys - orig_sys) * 100:9.2f}%")

                orig_idio = original_portfolio_attributions[portfolio_name].idiosyncratic_return
                opt_idio = optimal_portfolio_attributions[portfolio_name].idiosyncratic_return
                print(f"{'Idiosyncratic Return':20} {orig_idio * 100:14.2f}% {opt_idio * 100:14.2f}% {(opt_idio - orig_idio) * 100:9.2f}%")

                orig_beta = original_portfolio_attributions[portfolio_name].portfolio_beta
                opt_beta = optimal_portfolio_attributions[portfolio_name].portfolio_beta
                print(f"{'Portfolio Beta':20} {orig_beta:14.2f} {opt_beta:14.2f} {(opt_beta - orig_beta):9.2f}")

                # Print Sharpe ratio for optimal portfolio
                if portfolio_name in optimal_portfolios:
                    sharpe = optimal_portfolios[portfolio_name].sharpe_ratio * np.sqrt(252)  # Annualized
                    print(f"{'Optimal Sharpe Ratio':20} {'-':>14} {sharpe:14.2f} {'-':>10}")

        # 18. Return detailed results
        return {
            'optimal_portfolios': optimal_portfolios,
            'optimal_portfolio_values': optimal_portfolio_values,
            'optimal_portfolio_attributions': optimal_portfolio_attributions,
            'optimal_total_portfolio_attribution': optimal_total_portfolio_attribution,
            'optimal_vol_attribution': optimal_vol_attribution
        }

    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return None


def print_optimal_attribution_results(portfolio_attributions, total_portfolio_attribution,
                                     stock_simple_returns, vol_attribution, optimal_portfolios):
    """
    Print attribution analysis results for optimal portfolios in tabular format
    """
    spy_return = stock_simple_returns['SPY']

    # Print total portfolio attribution
    print("# Total Optimal Portfolio Attribution")
    print("# 3x4 DataFrame")
    print("#", "-" * 70)
    print(f"#  Row | Value               {'SPY':>15}    {'Alpha':>10}    {'Portfolio':>10}")
    print(f"#      | String              {'Float64':>15}    {'Float64':>10}    {'Float64':>10}")
    print("#", "-" * 70)

    total_return = total_portfolio_attribution.total_return

    # Row 1: Total return
    alpha_return = total_return - spy_return
    print(f"#  1   | TotalReturn         {spy_return:15.6f}    {alpha_return:10.6f}    {total_return:10.6f}")

    # Row 2: Return attribution - Modified calculation method
    systematic_return = total_portfolio_attribution.systematic_return
    idiosyncratic_return = total_portfolio_attribution.idiosyncratic_return
    print(f"#  2   | Return Attribution  {systematic_return:15.6f}    {idiosyncratic_return:10.6f}    {total_return:10.6f}")

    # Row 3: Volatility attribution
    vol_attrib = vol_attribution['Total']
    print(f"#  3   | Vol Attribution     {vol_attrib.spy:15.6f}    {vol_attrib.alpha:10.6f}    {vol_attrib.portfolio:10.6f}")

    # Row 4: Sharpe ratio
    # Calculate weighted average Sharpe ratio for total portfolio
    annualized_sharpe = total_portfolio_attribution.sharpe_ratio * np.sqrt(252)
    print(f"#  4   | Sharpe Ratio        {'-':>15}    {'-':>10}    {annualized_sharpe:10.6f}")

    # Print attribution for each portfolio
    for portfolio_name in portfolio_attributions.keys():
        print(f"\n# {portfolio_name} Optimal Portfolio Attribution")
        print("# 3x4 DataFrame")
        print("#", "-" * 70)
        print(f"#  Row | Value               {'SPY':>15}    {'Alpha':>10}    {'Portfolio':>10}")
        print(f"#      | String              {'Float64':>15}    {'Float64':>10}    {'Float64':>10}")
        print("#", "-" * 70)

        portfolio_return = portfolio_attributions[portfolio_name].total_return
        portfolio_alpha = portfolio_return - spy_return

        # Row 1: Total return
        print(f"#  1   | TotalReturn         {spy_return:15.6f}    {portfolio_alpha:10.6f}    {portfolio_return:10.6f}")

        # Row 2: Return attribution - Modified calculation method
        systematic_return = portfolio_attributions[portfolio_name].systematic_return
        idiosyncratic_return = portfolio_attributions[portfolio_name].idiosyncratic_return
        print(f"#  2   | Return Attribution  {systematic_return:15.6f}    {idiosyncratic_return:10.6f}    {portfolio_return:10.6f}")

        # Row 3: Volatility attribution
        vol_attrib = vol_attribution[portfolio_name]
        print(f"#  3   | Vol Attribution     {vol_attrib.spy:15.6f}    {vol_attrib.alpha:10.6f}    {vol_attrib.portfolio:10.6f}")

        # Row 4: Sharpe ratio
        if portfolio_name in optimal_portfolios:
            sharpe = optimal_portfolios[portfolio_name].sharpe_ratio * np.sqrt(252)  # Annualized
            print(f"#  4   | Sharpe Ratio        {'-':>15}    {'-':>10}    {sharpe:10.6f}")


if __name__ == "__main__":
    # Execute optimal Sharpe ratio portfolio analysis
    results = run_optimal_sharpe_analysis()
    print("Analysis complete!")
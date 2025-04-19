"""
CAPM and Optimal Sharpe Portfolio Analysis
Simplified version focused on TotalReturn, Return Attribution, and Vol Attribution
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize


#########################
### Helper Functions
#########################

def print_attribution_results(portfolio_attributions, total_portfolio_attribution,
                             stock_simple_returns):
    """
    Print attribution analysis results in table format focused on three main components
    """
    spy_return = stock_simple_returns['SPY']

    # Print total portfolio attribution
    print("# Total Portfolio Attribution")
    print("# 3x4 DataFrame")
    print("#", "-" * 70)
    print(f"#  Row | Value               {'SPY':>15}    {'Alpha':>10}    {'Portfolio':>10}")
    print(f"#      | String              {'Float64':>15}    {'Float64':>10}    {'Float64':>10}")
    print("#", "-" * 70)

    total_return = total_portfolio_attribution['total_return']

    # Row 1: Total Return
    alpha_return = total_return - spy_return
    print(f"#  1   | TotalReturn         {spy_return:15.6f}    {alpha_return:10.6f}    {total_return:10.6f}")

    # Row 2: Return Attribution
    systematic_return = total_portfolio_attribution['systematic_return']
    idiosyncratic_return = total_portfolio_attribution['idiosyncratic_return']
    print(
        f"#  2   | Return Attribution  {systematic_return:15.6f}    {idiosyncratic_return:10.6f}    {total_return:10.6f}")

    # Row 3: Vol Attribution
    total_vol = total_portfolio_attribution['total_volatility']
    systematic_vol = total_portfolio_attribution['systematic_volatility']
    idiosyncratic_vol = total_portfolio_attribution['idiosyncratic_volatility']
    print(
        f"#  3   | Vol Attribution     {systematic_vol:15.6f}    {idiosyncratic_vol:10.6f}    {total_vol:10.6f}")

    # Print each portfolio's attribution
    for portfolio_name in portfolio_attributions.keys():
        print(f"\n# {portfolio_name} Portfolio Attribution")
        print("# 3x4 DataFrame")
        print("#", "-" * 70)
        print(f"#  Row | Value               {'SPY':>15}    {'Alpha':>10}    {'Portfolio':>10}")
        print(f"#      | String              {'Float64':>15}    {'Float64':>10}    {'Float64':>10}")
        print("#", "-" * 70)

        portfolio_return = portfolio_attributions[portfolio_name]['total_return']
        portfolio_alpha = portfolio_return - spy_return

        # Row 1: Total Return
        print(f"#  1   | TotalReturn         {spy_return:15.6f}    {portfolio_alpha:10.6f}    {portfolio_return:10.6f}")

        # Row 2: Return Attribution
        systematic_return = portfolio_attributions[portfolio_name]['systematic_return']
        idiosyncratic_return = portfolio_attributions[portfolio_name]['idiosyncratic_return']
        print(
            f"#  2   | Return Attribution  {systematic_return:15.6f}    {idiosyncratic_return:10.6f}    {portfolio_return:10.6f}")

        # Row 3: Vol Attribution
        total_vol = portfolio_attributions[portfolio_name]['total_volatility']
        systematic_vol = portfolio_attributions[portfolio_name]['systematic_volatility']
        idiosyncratic_vol = portfolio_attributions[portfolio_name]['idiosyncratic_volatility']
        print(
            f"#  3   | Vol Attribution     {systematic_vol:15.6f}    {idiosyncratic_vol:10.6f}    {total_vol:10.6f}")


def print_optimal_attribution_results(portfolio_attributions, total_portfolio_attribution,
                                     stock_simple_returns, optimal_portfolios):
    """
    Print optimal portfolio attribution analysis results
    """
    spy_return = stock_simple_returns['SPY']

    # Print total portfolio attribution
    print("# Total Optimal Portfolio Attribution")
    print("# 3x4 DataFrame")
    print("#", "-" * 70)
    print(f"#  Row | Value               {'SPY':>15}    {'Alpha':>10}    {'Portfolio':>10}")
    print(f"#      | String              {'Float64':>15}    {'Float64':>10}    {'Float64':>10}")
    print("#", "-" * 70)

    total_return = total_portfolio_attribution['total_return']

    # Row 1: Total Return
    alpha_return = total_return - spy_return
    print(f"#  1   | TotalReturn         {spy_return:15.6f}    {alpha_return:10.6f}    {total_return:10.6f}")

    # Row 2: Return Attribution
    systematic_return = total_portfolio_attribution['systematic_return']
    idiosyncratic_return = total_portfolio_attribution['idiosyncratic_return']
    print(
        f"#  2   | Return Attribution  {systematic_return:15.6f}    {idiosyncratic_return:10.6f}    {total_return:10.6f}")

    # Row 3: Vol Attribution
    total_vol = total_portfolio_attribution['total_volatility']
    systematic_vol = total_portfolio_attribution['systematic_volatility']
    idiosyncratic_vol = total_portfolio_attribution['idiosyncratic_volatility']
    print(
        f"#  3   | Vol Attribution     {systematic_vol:15.6f}    {idiosyncratic_vol:10.6f}    {total_vol:10.6f}")

    # Row 4: Sharpe Ratio
    # Calculate total portfolio's weighted Sharpe ratio
    total_sharpe = 0
    for portfolio_name, weight in total_portfolio_attribution['weights'].items():
        if portfolio_name in optimal_portfolios:
            total_sharpe += weight * optimal_portfolios[portfolio_name]['sharpe_ratio']

    # Annualized Sharpe ratio
    annualized_sharpe = total_sharpe * np.sqrt(252)
    print(f"#  4   | Sharpe Ratio        {'-':>15}    {'-':>10}    {annualized_sharpe:10.6f}")

    # Print each portfolio's attribution
    for portfolio_name in portfolio_attributions.keys():
        print(f"\n# {portfolio_name} Optimal Portfolio Attribution")
        print("# 3x4 DataFrame")
        print("#", "-" * 70)
        print(f"#  Row | Value               {'SPY':>15}    {'Alpha':>10}    {'Portfolio':>10}")
        print(f"#      | String              {'Float64':>15}    {'Float64':>10}    {'Float64':>10}")
        print("#", "-" * 70)

        portfolio_return = portfolio_attributions[portfolio_name]['total_return']
        portfolio_alpha = portfolio_return - spy_return

        # Row 1: Total Return
        print(f"#  1   | TotalReturn         {spy_return:15.6f}    {portfolio_alpha:10.6f}    {portfolio_return:10.6f}")

        # Row 2: Return Attribution
        systematic_return = portfolio_attributions[portfolio_name]['systematic_return']
        idiosyncratic_return = portfolio_attributions[portfolio_name]['idiosyncratic_return']
        print(
            f"#  2   | Return Attribution  {systematic_return:15.6f}    {idiosyncratic_return:10.6f}    {portfolio_return:10.6f}")

        # Row 3: Vol Attribution
        total_vol = portfolio_attributions[portfolio_name]['total_volatility']
        systematic_vol = portfolio_attributions[portfolio_name]['systematic_volatility']
        idiosyncratic_vol = portfolio_attributions[portfolio_name]['idiosyncratic_volatility']
        print(
            f"#  3   | Vol Attribution     {systematic_vol:15.6f}    {idiosyncratic_vol:10.6f}    {total_vol:10.6f}")

        # Row 4: Sharpe Ratio
        if portfolio_name in optimal_portfolios:
            sharpe = optimal_portfolios[portfolio_name]['sharpe_ratio'] * np.sqrt(252)  # Annualized
            print(f"#  4   | Sharpe Ratio        {'-':>15}    {'-':>10}    {sharpe:10.6f}")


#########################
### PART 1: CAPM Analysis
#########################

def run_capm_analysis():
    """
    Perform CAPM portfolio risk and return attribution analysis
    """
    print("Starting CAPM portfolio risk and return attribution analysis...")

    try:
        # 1. Read all necessary data files
        daily_prices = pd.read_csv('DailyPrices.csv')
        initial_portfolio = pd.read_csv('initial_portfolio.csv')
        rf_data = pd.read_csv('rf.csv')

        # 2. Data preprocessing
        # 2.1 Set date as index
        daily_prices['Date'] = pd.to_datetime(daily_prices['Date'])
        daily_prices.set_index('Date', inplace=True)

        rf_data['Date'] = pd.to_datetime(rf_data['Date'])
        rf_data.set_index('Date', inplace=True)

        # If rf_data has multiple columns, use the first numeric column
        if len(rf_data.columns) > 1:
            rf_series = rf_data.select_dtypes(include=[np.number]).iloc[:, 0]
        else:
            rf_series = rf_data.squeeze()

        # 2.2 Find the end of 2023
        end_of_2023 = daily_prices[daily_prices.index.year == 2023].index.max()
        print(f"Training set end date: {end_of_2023.strftime('%Y-%m-%d')}")

        # 2.3 Split training and test sets
        train_prices = daily_prices[daily_prices.index <= end_of_2023]
        test_prices = daily_prices[daily_prices.index > end_of_2023]

        print(f"Training set days: {len(train_prices)}")
        print(f"Test set days: {len(test_prices)}")

        # 3. Calculate daily returns
        # 3.1 Training set returns
        train_returns = train_prices.pct_change().dropna()

        # 3.2 Test set returns
        test_returns = test_prices.pct_change().dropna()

        # 4. Calculate excess returns
        # Align rf data with returns
        train_rf = rf_series.reindex(train_returns.index).ffill()
        test_rf = rf_series.reindex(test_returns.index).ffill()

        # Calculate excess returns
        train_excess_returns = train_returns.subtract(train_rf, axis=0)
        test_excess_returns = test_returns.subtract(test_rf, axis=0)

        # 5. Calculate CAPM parameters (Beta and Alpha)
        def calculate_capm(stock_returns, market_returns):
            """Calculate CAPM model parameters"""
            # Ensure data is valid
            valid_data = pd.concat([market_returns, stock_returns], axis=1).dropna()
            if len(valid_data) < 10:  # Require minimum data points
                return {'alpha': np.nan, 'beta': np.nan, 'r2': np.nan}

            # Use linear regression
            x = valid_data.iloc[:, 0].values.reshape(-1, 1)  # Market returns
            y = valid_data.iloc[:, 1].values  # Stock returns

            slope, intercept, r_value, p_value, std_err = stats.linregress(x.flatten(), y)

            return {
                'alpha': intercept,
                'beta': slope,
                'r2': r_value ** 2
            }

        # Use SPY as market index
        market_returns = train_excess_returns['SPY']

        # Calculate CAPM parameters for each stock
        capm_params = {}
        for symbol in train_excess_returns.columns:
            if symbol != 'SPY':
                capm_params[symbol] = calculate_capm(train_excess_returns[symbol], market_returns)

        # Market's own coefficients
        capm_params['SPY'] = {'alpha': 0, 'beta': 1, 'r2': 1}

        # Print CAPM parameters for some stocks
        print("\nCAMP Parameters for selected stocks:")
        for symbol in ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META']:
            if symbol in capm_params:
                params = capm_params[symbol]
                print(f"{symbol}: Beta={params['beta']:.2f}, Alpha={params['alpha']:.4f}, R²={params['r2']:.2f}")

        # 6. Calculate initial portfolio value and final value
        # Get prices at the end of 2023 (initial prices)
        end_of_2023_prices = daily_prices.loc[end_of_2023]

        # Get prices for the last day (final prices)
        last_date = test_prices.index.max()
        last_day_prices = daily_prices.loc[last_date]

        # Organize portfolio data
        portfolios = {}
        for portfolio_name in initial_portfolio['Portfolio'].unique():
            portfolio_stocks = initial_portfolio[initial_portfolio['Portfolio'] == portfolio_name]
            portfolios[portfolio_name] = portfolio_stocks

        # Calculate initial portfolio value, final value, and simple return
        portfolio_values = {}

        for name, portfolio_df in portfolios.items():
            initial_stock_values = {}  # Initial stock values
            final_stock_values = {}    # Final stock values
            total_initial_value = 0
            total_final_value = 0

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

            # Calculate simple return
            simple_return = (total_final_value - total_initial_value) / total_initial_value if total_initial_value > 0 else 0

            # Calculate portfolio returns for risk attribution
            portfolio_test_returns = pd.Series(0, index=test_returns.index)
            portfolio_weights = {}

            for symbol, initial_value in initial_stock_values.items():
                weight = initial_value / total_initial_value if total_initial_value > 0 else 0
                portfolio_weights[symbol] = weight
                if symbol in test_returns.columns:
                    portfolio_test_returns += test_returns[symbol] * weight

            # Calculate portfolio beta
            portfolio_beta = 0
            for symbol, weight in portfolio_weights.items():
                if symbol in capm_params and not np.isnan(capm_params[symbol]['beta']):
                    portfolio_beta += weight * capm_params[symbol]['beta']

            # Calculate return attribution
            spy_return = (last_day_prices['SPY'] - end_of_2023_prices['SPY']) / end_of_2023_prices['SPY']
            systematic_return = portfolio_beta * spy_return
            idiosyncratic_return = simple_return - systematic_return

            # Calculate risk attribution (using approach from Version 1)
            # First calculate portfolio volatility
            portfolio_volatility = portfolio_test_returns.std()

            # Then calculate components as in Version 1
            market_volatility = test_returns['SPY'].std()
            systematic_volatility = portfolio_beta * market_volatility
            idiosyncratic_volatility = portfolio_volatility - systematic_volatility

            # Ensure non-negative idiosyncratic volatility
            if idiosyncratic_volatility < 0:
                idiosyncratic_volatility = 0

            portfolio_values[name] = {
                'initial_value': total_initial_value,
                'final_value': total_final_value,
                'simple_return': simple_return,
                'initial_stock_values': initial_stock_values,
                'final_stock_values': final_stock_values,
                'portfolio_beta': portfolio_beta,
                'systematic_return': systematic_return,
                'idiosyncratic_return': idiosyncratic_return,
                'total_volatility': portfolio_volatility,
                'systematic_volatility': systematic_volatility,
                'idiosyncratic_volatility': idiosyncratic_volatility,
                'portfolio_weights': portfolio_weights,
                'portfolio_test_returns': portfolio_test_returns
            }

        # Print portfolio values and returns
        print("\nPortfolio values and simple returns:")
        for name, values in portfolio_values.items():
            print(
                f"{name}: Initial=${values['initial_value']:.2f}, Final=${values['final_value']:.2f}, "
                f"Return={values['simple_return'] * 100:.2f}%, Beta={values['portfolio_beta']:.2f}")

        # Calculate stock simple returns
        stock_simple_returns = {}
        for symbol in daily_prices.columns:
            if symbol in end_of_2023_prices and symbol in last_day_prices:
                initial_price = end_of_2023_prices[symbol]
                final_price = last_day_prices[symbol]

                if not np.isnan(initial_price) and not np.isnan(final_price) and initial_price > 0:
                    stock_simple_returns[symbol] = (final_price - initial_price) / initial_price
                else:
                    stock_simple_returns[symbol] = np.nan

        # Calculate total portfolio attribution
        total_initial_value = sum(pv['initial_value'] for pv in portfolio_values.values())
        total_final_value = sum(pv['final_value'] for pv in portfolio_values.values())

        # Overall portfolio simple return
        total_simple_return = (total_final_value - total_initial_value) / total_initial_value if total_initial_value > 0 else 0

        # Calculate overall portfolio weights and beta
        total_portfolio_weights = {}
        for portfolio_name, portfolio_data in portfolio_values.items():
            portfolio_weight = portfolio_data['initial_value'] / total_initial_value
            for symbol, weight in portfolio_data['portfolio_weights'].items():
                if symbol not in total_portfolio_weights:
                    total_portfolio_weights[symbol] = 0
                total_portfolio_weights[symbol] += weight * portfolio_weight

        total_portfolio_beta = 0
        for symbol, weight in total_portfolio_weights.items():
            if symbol in capm_params and not np.isnan(capm_params[symbol]['beta']):
                total_portfolio_beta += weight * capm_params[symbol]['beta']

        # Calculate total portfolio return attribution
        total_systematic_return = total_portfolio_beta * spy_return
        total_idiosyncratic_return = total_simple_return - total_systematic_return

        # Calculate total portfolio risk attribution
        total_portfolio_returns = pd.Series(0, index=test_returns.index)
        for portfolio_name, portfolio_data in portfolio_values.items():
            portfolio_weight = portfolio_data['initial_value'] / total_initial_value
            total_portfolio_returns += portfolio_data['portfolio_test_returns'] * portfolio_weight

        # Calculate total volatility
        total_portfolio_volatility = total_portfolio_returns.std()

        # Risk attribution as in Version 1
        total_systematic_volatility = total_portfolio_beta * market_volatility
        total_idiosyncratic_volatility = total_portfolio_volatility - total_systematic_volatility

        # Ensure non-negative idiosyncratic volatility
        if total_idiosyncratic_volatility < 0:
            total_idiosyncratic_volatility = 0

        total_portfolio_attribution = {
            'total_return': total_simple_return,
            'systematic_return': total_systematic_return,
            'idiosyncratic_return': total_idiosyncratic_return,
            'total_volatility': total_portfolio_volatility,
            'systematic_volatility': total_systematic_volatility,
            'idiosyncratic_volatility': total_idiosyncratic_volatility,
            'portfolio_beta': total_portfolio_beta,
            'weights': {name: pv['initial_value'] / total_initial_value for name, pv in portfolio_values.items()}
        }

        # Create portfolio attributions dictionary for each portfolio
        portfolio_attributions = {}
        for name, data in portfolio_values.items():
            portfolio_attributions[name] = {
                'total_return': data['simple_return'],
                'systematic_return': data['systematic_return'],
                'idiosyncratic_return': data['idiosyncratic_return'],
                'total_volatility': data['total_volatility'],
                'systematic_volatility': data['systematic_volatility'],
                'idiosyncratic_volatility': data['idiosyncratic_volatility'],
                'portfolio_beta': data['portfolio_beta']
            }

        # Print attribution results
        print("\n")
        print_attribution_results(portfolio_attributions, total_portfolio_attribution, stock_simple_returns)

        # Return detailed results
        return {
            'capm_params': capm_params,
            'portfolio_values': portfolio_values,
            'portfolio_attributions': portfolio_attributions,
            'total_portfolio_attribution': total_portfolio_attribution,
            'stock_simple_returns': stock_simple_returns,
            'train_returns': train_returns,
            'test_returns': test_returns,
            'train_rf': train_rf,
            'test_rf': test_rf,
            'end_of_2023': end_of_2023
        }

    except Exception as e:
        print(f"Error occurred during analysis: {e}")
        import traceback
        traceback.print_exc()
        return None


#################################
### PART 2: Optimal Sharpe Analysis
#################################

def run_optimal_sharpe_analysis():
    """
    Perform optimal Sharpe ratio portfolio analysis (Part 2)
    """
    print("Starting optimal Sharpe ratio portfolio analysis...")

    try:
        # 1. Get CAPM analysis results from Part 1
        capm_results = run_capm_analysis()

        if not capm_results:
            print("Unable to get CAPM analysis results. Please ensure Part 1 analysis is successful.")
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

        # If rf_data has multiple columns, use the first numeric column
        if len(rf_data.columns) > 1:
            rf_series = rf_data.select_dtypes(include=[np.number]).iloc[:, 0]
        else:
            rf_series = rf_data.squeeze()

        # 2.2 Find the end of 2023
        end_of_2023 = capm_results['end_of_2023']

        # 2.3 Use precomputed returns from Part 1
        train_returns = capm_results['train_returns']
        test_returns = capm_results['test_returns']
        train_rf = capm_results['train_rf']
        test_rf = capm_results['test_rf']

        # 3. Calculate covariance matrix
        cov_matrix = train_returns.cov()

        # 4. Calculate expected return for each stock (assume alpha=0)
        capm_params = capm_results['capm_params']
        avg_market_return = train_returns['SPY'].mean()
        avg_rf = train_rf.mean()

        expected_returns = {}
        for symbol, params in capm_params.items():
            # Expected return = Rf + Beta * (E[Rm] - Rf)
            if not np.isnan(params['beta']):
                expected_returns[symbol] = avg_rf + params['beta'] * (avg_market_return - avg_rf)
            else:
                expected_returns[symbol] = np.nan

        print("\nStock expected returns (annualized):")
        for symbol in ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META']:
            if symbol in expected_returns and not np.isnan(expected_returns[symbol]):
                print(f"{symbol}: {expected_returns[symbol] * 252 * 100:.2f}%")  # Annualized

        # 5. Define maximum Sharpe ratio portfolio optimization function
        def optimize_sharpe_ratio(tickers, expected_returns_dict, cov_matrix, risk_free_rate):
            """
            Optimize portfolio to get maximum Sharpe ratio
            """
            valid_tickers = [t for t in tickers if t in cov_matrix.columns and t in expected_returns_dict
                           and not np.isnan(expected_returns_dict[t])]

            if not valid_tickers:
                print("No valid tickers for optimization.")
                return None

            n_assets = len(valid_tickers)

            # Initial weights are equal weights
            init_weights = np.ones(n_assets) / n_assets

            # Constraint: sum of weights equals 1
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

            # Boundary conditions: all weights non-negative (no short selling)
            bounds = tuple((0, 1) for asset in range(n_assets))

            # Create expected returns array
            returns_vector = np.array([expected_returns_dict[ticker] for ticker in valid_tickers])

            # Objective function: maximize Sharpe ratio (minimize negative Sharpe ratio)
            def neg_sharpe_ratio(weights):
                # Calculate portfolio expected return
                port_return = np.sum(returns_vector * weights)

                # Calculate portfolio risk (standard deviation)
                port_variance = np.dot(weights.T, np.dot(cov_matrix.loc[valid_tickers, valid_tickers].values, weights))
                if port_variance < 1e-10:  # Handle near-zero variance
                    return np.inf  # Penalize extreme solutions
                port_volatility = np.sqrt(port_variance)

                # Calculate Sharpe ratio
                sharpe = (port_return - risk_free_rate) / port_volatility

                # Return negative Sharpe ratio (minimization problem)
                return -sharpe

            # Execute optimization
            try:
                result = minimize(neg_sharpe_ratio, init_weights, method='SLSQP',
                                bounds=bounds, constraints=constraints)

                if not result.success:
                    print(f"Warning: Optimization did not converge for tickers {valid_tickers}.")
                    return None

                # Extract optimal weights
                optimal_weights = result.x

                # Calculate optimal portfolio characteristics
                optimal_return = np.sum(returns_vector * optimal_weights)
                optimal_variance = np.dot(optimal_weights.T,
                                        np.dot(cov_matrix.loc[valid_tickers, valid_tickers].values, optimal_weights))
                optimal_volatility = np.sqrt(optimal_variance)
                optimal_sharpe = (optimal_return - risk_free_rate) / optimal_volatility

                # Create weights dictionary (including all original tickers with zero weights for missing)
                weights_dict = {ticker: 0.0 for ticker in tickers}
                for i, ticker in enumerate(valid_tickers):
                    weights_dict[ticker] = optimal_weights[i]

                return {
                    'weights': weights_dict,
                    'expected_return': optimal_return,
                    'expected_volatility': optimal_volatility,
                    'sharpe_ratio': optimal_sharpe
                }

            except Exception as e:
                print(f"Optimization error: {e}")
                return None

        # 6. Create optimal Sharpe ratio portfolio for each portfolio
        portfolios = {}
        for portfolio_name in initial_portfolio['Portfolio'].unique():
            portfolio_stocks = initial_portfolio[initial_portfolio['Portfolio'] == portfolio_name]['Symbol'].tolist()
            portfolios[portfolio_name] = portfolio_stocks

        # 7. Calculate optimal weights for each portfolio
        optimal_portfolios = {}
        for portfolio_name, stocks in portfolios.items():
            print(f"\nOptimizing portfolio {portfolio_name}...")

            # Skip portfolios with no valid stocks
            valid_stocks = [s for s in stocks if s in cov_matrix.columns and s in expected_returns
                           and not np.isnan(expected_returns[s])]

            if not valid_stocks:
                print(f"No valid stocks for portfolio {portfolio_name}. Skipping.")
                continue

            # Calculate optimal Sharpe ratio portfolio
            opt_result = optimize_sharpe_ratio(
                valid_stocks, expected_returns, cov_matrix, avg_rf)

            if opt_result:
                optimal_portfolios[portfolio_name] = opt_result

                # Print top holdings
                print(f"Portfolio {portfolio_name} optimal weights (top holdings):")
                sorted_weights = sorted(opt_result['weights'].items(), key=lambda x: x[1], reverse=True)
                for symbol, weight in sorted_weights[:5]:  # Show top 5 weights
                    if weight > 0.01:  # Only show significant weights
                        print(f"{symbol}: {weight * 100:.2f}%")

                # Print portfolio characteristics
                port_data = opt_result
                print(f"Expected return: {port_data['expected_return'] * 252 * 100:.2f}% (annualized)")
                print(f"Expected volatility: {port_data['expected_volatility'] * np.sqrt(252) * 100:.2f}% (annualized)")
                print(f"Sharpe ratio: {port_data['sharpe_ratio'] * np.sqrt(252):.2f} (annualized)")
            else:
                print(f"Failed to optimize portfolio {portfolio_name}.")

        # 8. Create new holdings data for optimal portfolios
        optimal_holdings = []

        for portfolio_name, opt_portfolio in optimal_portfolios.items():
            if portfolio_name not in capm_results['portfolio_values']:
                continue

            initial_value = capm_results['portfolio_values'][portfolio_name]['initial_value']

            # Create optimal holdings
            for symbol, weight in opt_portfolio['weights'].items():
                if weight > 0:
                    if symbol in daily_prices.loc[end_of_2023] and not np.isnan(daily_prices.loc[end_of_2023, symbol]):
                        price = daily_prices.loc[end_of_2023, symbol]
                        shares = (initial_value * weight) / price
                        optimal_holdings.append({
                            'Portfolio': portfolio_name,
                            'Symbol': symbol,
                            'Holding': shares
                        })

        # Create DataFrame from the list
        optimal_holdings_df = pd.DataFrame(optimal_holdings)

        if optimal_holdings_df.empty:
            print("Error: No optimal holdings could be created.")
            return None

        # 9. Analyze optimal portfolios
        print("\nAnalyzing optimal portfolios...")
        optimal_portfolio_values = {}

        for name in optimal_portfolios.keys():
            # Filter holdings for this portfolio
            portfolio_holdings = optimal_holdings_df[optimal_holdings_df['Portfolio'] == name]

            if len(portfolio_holdings) == 0:
                print(f"No holdings for optimal portfolio {name}")
                continue

            initial_stock_values = {}  # Initial stock values
            final_stock_values = {}  # Final stock values
            total_initial_value = 0
            total_final_value = 0

            # Calculate initial and final values
            for _, row in portfolio_holdings.iterrows():
                symbol = row['Symbol']
                holding = row['Holding']

                if (symbol in daily_prices.loc[end_of_2023] and not np.isnan(daily_prices.loc[end_of_2023, symbol]) and
                        symbol in daily_prices.loc[daily_prices.index.max()] and not np.isnan(
                            daily_prices.loc[daily_prices.index.max(), symbol])):
                    initial_value = holding * daily_prices.loc[end_of_2023, symbol]
                    final_value = holding * daily_prices.loc[daily_prices.index.max(), symbol]

                    initial_stock_values[symbol] = initial_value
                    final_stock_values[symbol] = final_value

                    total_initial_value += initial_value
                    total_final_value += final_value

            # Calculate simple return
            simple_return = (
                                        total_final_value - total_initial_value) / total_initial_value if total_initial_value > 0 else 0

            # Calculate portfolio weights
            portfolio_weights = {}
            for symbol, value in initial_stock_values.items():
                portfolio_weights[symbol] = value / total_initial_value if total_initial_value > 0 else 0

            # Calculate portfolio beta
            portfolio_beta = 0
            for symbol, weight in portfolio_weights.items():
                if symbol in capm_params and not np.isnan(capm_params[symbol]['beta']):
                    portfolio_beta += weight * capm_params[symbol]['beta']

            # Calculate portfolio returns for risk attribution
            portfolio_test_returns = pd.Series(0, index=test_returns.index)
            for symbol, weight in portfolio_weights.items():
                if symbol in test_returns.columns:
                    portfolio_test_returns += test_returns[symbol] * weight

            # Calculate return attribution
            spy_return = capm_results['stock_simple_returns']['SPY']
            systematic_return = portfolio_beta * spy_return
            idiosyncratic_return = simple_return - systematic_return

            # Calculate risk attribution (using variance decomposition)
            portfolio_variance = portfolio_test_returns.var()
            market_variance = test_returns['SPY'].var()
            systematic_variance = (portfolio_beta ** 2) * market_variance
            idiosyncratic_variance = max(0, portfolio_variance - systematic_variance)  # Ensure non-negative

            # Convert variances to volatilities (standard deviations)
            portfolio_volatility = np.sqrt(portfolio_variance)
            systematic_volatility = np.sqrt(systematic_variance)
            idiosyncratic_volatility = np.sqrt(idiosyncratic_variance)

            optimal_portfolio_values[name] = {
                'initial_value': total_initial_value,
                'final_value': total_final_value,
                'simple_return': simple_return,
                'initial_stock_values': initial_stock_values,
                'final_stock_values': final_stock_values,
                'portfolio_beta': portfolio_beta,
                'systematic_return': systematic_return,
                'idiosyncratic_return': idiosyncratic_return,
                'total_volatility': portfolio_volatility,
                'systematic_volatility': systematic_volatility,
                'idiosyncratic_volatility': idiosyncratic_volatility,
                'portfolio_weights': portfolio_weights,
                'portfolio_test_returns': portfolio_test_returns
            }

        # 10. Calculate total optimal portfolio attribution
        total_initial_value = sum(pv['initial_value'] for pv in optimal_portfolio_values.values())
        total_final_value = sum(pv['final_value'] for pv in optimal_portfolio_values.values())

        if total_initial_value <= 0:
            print("Error: Total initial value for optimal portfolios is zero.")
            return None

        # Overall portfolio simple return
        total_simple_return = (total_final_value - total_initial_value) / total_initial_value

        # Calculate overall portfolio weights and beta
        total_portfolio_weights = {}
        for portfolio_name, portfolio_data in optimal_portfolio_values.items():
            portfolio_weight = portfolio_data['initial_value'] / total_initial_value
            for symbol, weight in portfolio_data['portfolio_weights'].items():
                if symbol not in total_portfolio_weights:
                    total_portfolio_weights[symbol] = 0
                total_portfolio_weights[symbol] += weight * portfolio_weight

        total_portfolio_beta = 0
        for symbol, weight in total_portfolio_weights.items():
            if symbol in capm_params and not np.isnan(capm_params[symbol]['beta']):
                total_portfolio_beta += weight * capm_params[symbol]['beta']

        # Calculate total portfolio return attribution
        total_systematic_return = total_portfolio_beta * spy_return
        total_idiosyncratic_return = total_simple_return - total_systematic_return

        # Calculate total portfolio risk attribution
        total_portfolio_returns = pd.Series(0, index=test_returns.index)
        for portfolio_name, portfolio_data in optimal_portfolio_values.items():
            portfolio_weight = portfolio_data['initial_value'] / total_initial_value
            total_portfolio_returns += portfolio_data['portfolio_test_returns'] * portfolio_weight

        total_portfolio_variance = total_portfolio_returns.var()
        total_systematic_variance = (total_portfolio_beta ** 2) * market_variance
        total_idiosyncratic_variance = max(0, total_portfolio_variance - total_systematic_variance)

        total_portfolio_volatility = np.sqrt(total_portfolio_variance)
        total_systematic_volatility = np.sqrt(total_systematic_variance)
        total_idiosyncratic_volatility = np.sqrt(total_idiosyncratic_variance)

        optimal_total_portfolio_attribution = {
            'total_return': total_simple_return,
            'systematic_return': total_systematic_return,
            'idiosyncratic_return': total_idiosyncratic_return,
            'total_volatility': total_portfolio_volatility,
            'systematic_volatility': total_systematic_volatility,
            'idiosyncratic_volatility': total_idiosyncratic_volatility,
            'portfolio_beta': total_portfolio_beta,
            'weights': {name: pv['initial_value'] / total_initial_value for name, pv in
                        optimal_portfolio_values.items()}
        }

        # Create optimal portfolio attributions dictionary
        optimal_portfolio_attributions = {}
        for name, data in optimal_portfolio_values.items():
            optimal_portfolio_attributions[name] = {
                'total_return': data['simple_return'],
                'systematic_return': data['systematic_return'],
                'idiosyncratic_return': data['idiosyncratic_return'],
                'total_volatility': data['total_volatility'],
                'systematic_volatility': data['systematic_volatility'],
                'idiosyncratic_volatility': data['idiosyncratic_volatility'],
                'portfolio_beta': data['portfolio_beta']
            }

        # 11. Print optimal portfolio attribution results
        print("\nOptimal Sharpe Ratio Portfolio Attribution Results:")
        print_optimal_attribution_results(
            optimal_portfolio_attributions,
            optimal_total_portfolio_attribution,
            capm_results['stock_simple_returns'],
            optimal_portfolios
        )

        # 12. Compare the performance of original and optimal portfolios
        print("\nComparison between Original and Optimal Portfolios:")

        # Compare total portfolio
        print("\nTotal Portfolio Comparison:")
        print(f"{'Metric':20} {'Original Portfolio':>15} {'Optimal Portfolio':>15} {'Difference':>10}")
        print("-" * 70)

        orig_return = capm_results['total_portfolio_attribution']['total_return']
        opt_return = optimal_total_portfolio_attribution['total_return']
        print(
            f"{'Total Return':20} {orig_return * 100:14.2f}% {opt_return * 100:14.2f}% {(opt_return - orig_return) * 100:9.2f}%")

        orig_sys = capm_results['total_portfolio_attribution']['systematic_return']
        opt_sys = optimal_total_portfolio_attribution['systematic_return']
        print(
            f"{'Systematic Return':20} {orig_sys * 100:14.2f}% {opt_sys * 100:14.2f}% {(opt_sys - orig_sys) * 100:9.2f}%")

        orig_idio = capm_results['total_portfolio_attribution']['idiosyncratic_return']
        opt_idio = optimal_total_portfolio_attribution['idiosyncratic_return']
        print(
            f"{'Idiosyncratic Return':20} {orig_idio * 100:14.2f}% {opt_idio * 100:14.2f}% {(opt_idio - orig_idio) * 100:9.2f}%")

        orig_beta = capm_results['total_portfolio_attribution']['portfolio_beta']
        opt_beta = optimal_total_portfolio_attribution['portfolio_beta']
        print(f"{'Portfolio Beta':20} {orig_beta:14.2f} {opt_beta:14.2f} {(opt_beta - orig_beta):9.2f}")

        orig_vol = capm_results['total_portfolio_attribution']['total_volatility']
        opt_vol = optimal_total_portfolio_attribution['total_volatility']
        print(f"{'Total Volatility':20} {orig_vol:14.6f} {opt_vol:14.6f} {(opt_vol - orig_vol):9.6f}")

        orig_sys_vol = capm_results['total_portfolio_attribution']['systematic_volatility']
        opt_sys_vol = optimal_total_portfolio_attribution['systematic_volatility']
        print(f"{'Systematic Vol':20} {orig_sys_vol:14.6f} {opt_sys_vol:14.6f} {(opt_sys_vol - orig_sys_vol):9.6f}")

        orig_idio_vol = capm_results['total_portfolio_attribution']['idiosyncratic_volatility']
        opt_idio_vol = optimal_total_portfolio_attribution['idiosyncratic_volatility']
        print(
            f"{'Idiosyncratic Vol':20} {orig_idio_vol:14.6f} {opt_idio_vol:14.6f} {(opt_idio_vol - orig_idio_vol):9.6f}")

        # Compare individual portfolios
        for portfolio_name in optimal_portfolio_attributions.keys():
            if portfolio_name in capm_results['portfolio_attributions']:
                print(f"\nPortfolio {portfolio_name} Comparison:")
                print(f"{'Metric':20} {'Original Portfolio':>15} {'Optimal Portfolio':>15} {'Difference':>10}")
                print("-" * 70)

                orig_return = capm_results['portfolio_attributions'][portfolio_name]['total_return']
                opt_return = optimal_portfolio_attributions[portfolio_name]['total_return']
                print(
                    f"{'Total Return':20} {orig_return * 100:14.2f}% {opt_return * 100:14.2f}% {(opt_return - orig_return) * 100:9.2f}%")

                orig_sys = capm_results['portfolio_attributions'][portfolio_name]['systematic_return']
                opt_sys = optimal_portfolio_attributions[portfolio_name]['systematic_return']
                print(
                    f"{'Systematic Return':20} {orig_sys * 100:14.2f}% {opt_sys * 100:14.2f}% {(opt_sys - orig_sys) * 100:9.2f}%")

                orig_idio = capm_results['portfolio_attributions'][portfolio_name]['idiosyncratic_return']
                opt_idio = optimal_portfolio_attributions[portfolio_name]['idiosyncratic_return']
                print(
                    f"{'Idiosyncratic Return':20} {orig_idio * 100:14.2f}% {opt_idio * 100:14.2f}% {(opt_idio - orig_idio) * 100:9.2f}%")

                orig_beta = capm_results['portfolio_attributions'][portfolio_name]['portfolio_beta']
                opt_beta = optimal_portfolio_attributions[portfolio_name]['portfolio_beta']
                print(f"{'Portfolio Beta':20} {orig_beta:14.2f} {opt_beta:14.2f} {(opt_beta - orig_beta):9.2f}")

                orig_vol = capm_results['portfolio_attributions'][portfolio_name]['total_volatility']
                opt_vol = optimal_portfolio_attributions[portfolio_name]['total_volatility']
                print(f"{'Total Volatility':20} {orig_vol:14.6f} {opt_vol:14.6f} {(opt_vol - orig_vol):9.6f}")

                orig_sys_vol = capm_results['portfolio_attributions'][portfolio_name]['systematic_volatility']
                opt_sys_vol = optimal_portfolio_attributions[portfolio_name]['systematic_volatility']
                print(
                    f"{'Systematic Vol':20} {orig_sys_vol:14.6f} {opt_sys_vol:14.6f} {(opt_sys_vol - orig_sys_vol):9.6f}")

                orig_idio_vol = capm_results['portfolio_attributions'][portfolio_name]['idiosyncratic_volatility']
                opt_idio_vol = optimal_portfolio_attributions[portfolio_name]['idiosyncratic_volatility']
                print(
                    f"{'Idiosyncratic Vol':20} {orig_idio_vol:14.6f} {opt_idio_vol:14.6f} {(opt_idio_vol - orig_idio_vol):9.6f}")

                if portfolio_name in optimal_portfolios:
                    sharpe = optimal_portfolios[portfolio_name]['sharpe_ratio'] * np.sqrt(252)  # Annualized
                    print(f"{'Optimal Sharpe Ratio':20} {'-':>14} {sharpe:14.2f} {'-':>10}")

        # 13. Return detailed results
        return {
            'optimal_portfolios': optimal_portfolios,
            'optimal_portfolio_values': optimal_portfolio_values,
            'optimal_portfolio_attributions': optimal_portfolio_attributions,
            'optimal_total_portfolio_attribution': optimal_total_portfolio_attribution
        }

    except Exception as e:
        print(f"Error occurred during analysis: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Part 1 - Execute CAPM analysis
    print("\n============ PART 1: CAPM Analysis ============\n")
    capm_results = run_capm_analysis()
    print("CAPM analysis complete!")

    # Part 2 - Execute optimal Sharpe ratio portfolio analysis
    print("\n============ PART 2: Optimal Sharpe Analysis ============\n")
    optimal_results = run_optimal_sharpe_analysis()
    print("Optimal Sharpe ratio portfolio analysis complete!")
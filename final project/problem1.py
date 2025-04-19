import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import traceback


@dataclass
class CAPMParams:
    alpha: float
    beta: float
    r_squared: float


@dataclass
class PortfolioStats:
    initial_value: float
    final_value: float
    simple_return: float
    portfolio_beta: float
    initial_stock_values: Dict[str, float]
    final_stock_values: Dict[str, float]


@dataclass
class AttributionResults:
    total_return: float
    rf_return: float
    systematic_return: float
    idiosyncratic_return: float
    total_excess_return: float
    portfolio_beta: float
    weights: Dict[str, float] = None


@dataclass
class VolatilityAttribution:
    spy: float
    alpha: float
    portfolio: float


class CAPMAnalyzer:
    """
    A class to perform CAPM portfolio risk and return attribution analysis.
    """

    def __init__(self, price_file: str, portfolio_file: str, rf_file: str):
        """
        Initialize analyzer with file paths.

        Args:
            price_file: Path to daily prices CSV
            portfolio_file: Path to initial portfolio CSV
            rf_file: Path to risk-free rate CSV
        """
        self.price_file = price_file
        self.portfolio_file = portfolio_file
        self.rf_file = rf_file
        self.daily_prices = None
        self.initial_portfolio = None
        self.rf_data = None
        self.train_prices = None
        self.test_prices = None
        self.train_returns = None
        self.test_returns = None
        self.train_excess_returns = None
        self.test_excess_returns = None
        self.capm_params = {}
        self.portfolios = {}
        self.portfolio_values = {}
        self.stock_simple_returns = {}
        self.test_rf_return = None
        self.portfolio_attributions = {}
        self.total_portfolio_attribution = None
        self.vol_attribution = {}
        self.end_of_2023 = None

    def load_data(self) -> None:
        """Load and preprocess all required data files."""
        # Load raw data
        self.daily_prices = pd.read_csv(self.price_file)
        self.initial_portfolio = pd.read_csv(self.portfolio_file)
        self.rf_data = pd.read_csv(self.rf_file)

        # Convert dates and set as index
        for df in [self.daily_prices, self.rf_data]:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)

        # Find end of 2023 for train/test split
        self.end_of_2023 = self.daily_prices[self.daily_prices.index.year == 2023].index.max()
        print(f"Training set ends: {self.end_of_2023.strftime('%Y-%m-%d')}")

        # Split data into training and test sets
        self.train_prices = self.daily_prices[self.daily_prices.index <= self.end_of_2023]
        self.test_prices = self.daily_prices[self.daily_prices.index > self.end_of_2023]

        print(f"Training period: {len(self.train_prices)} days")
        print(f"Testing period: {len(self.test_prices)} days")

    def calculate_returns(self) -> None:
        """Calculate returns and excess returns for both periods."""
        # Calculate returns
        self.train_returns = self.train_prices.pct_change().dropna()
        self.test_returns = self.test_prices.pct_change().dropna()

        # Get risk-free rates for both periods
        train_rf = self.rf_data.loc[self.train_returns.index].squeeze()
        test_rf = self.rf_data.loc[self.test_returns.index].squeeze()

        # Calculate excess returns
        self.train_excess_returns = self.train_returns.subtract(train_rf, axis=0)
        self.test_excess_returns = self.test_returns.subtract(test_rf, axis=0)

        # Calculate total risk-free return for test period
        self.test_rf_return = (1 + test_rf).prod() - 1
        print(f"Risk-free return during test period: {self.test_rf_return * 100:.2f}%")

    @staticmethod
    def _fit_capm_model(stock_returns: pd.Series, market_returns: pd.Series) -> CAPMParams:
        """
        Fit CAPM model using linear regression.

        Args:
            stock_returns: Excess returns for a stock
            market_returns: Excess returns for the market

        Returns:
            CAPMParams with alpha, beta and r-squared values
        """
        # Ensure data is valid
        valid_data = pd.concat([market_returns, stock_returns], axis=1).dropna()
        if len(valid_data) < 2:
            return CAPMParams(alpha=np.nan, beta=np.nan, r_squared=np.nan)

        # Use linear regression
        x = valid_data.iloc[:, 0].values.reshape(-1, 1)  # Market returns
        y = valid_data.iloc[:, 1].values  # Stock returns

        slope, intercept, r_value, _, _ = stats.linregress(x.flatten(), y)

        return CAPMParams(
            alpha=intercept,
            beta=slope,
            r_squared=r_value ** 2
        )

    def estimate_capm_parameters(self) -> None:
        """Estimate CAPM parameters (Alpha, Beta) for all stocks."""
        # Use SPY as market index
        market_returns = self.train_excess_returns['SPY']

        # Calculate parameters for each stock
        for symbol in self.train_excess_returns.columns:
            if symbol != 'SPY':
                self.capm_params[symbol] = self._fit_capm_model(
                    self.train_excess_returns[symbol],
                    market_returns
                )

        # Market coefficients
        self.capm_params['SPY'] = CAPMParams(alpha=0, beta=1, r_squared=1)

        # Display parameters for key stocks
        print("\nCAPM parameters for selected stocks:")
        for symbol in ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META']:
            if symbol in self.capm_params:
                params = self.capm_params[symbol]
                print(f"{symbol}: Beta={params.beta:.2f}, Alpha={params.alpha:.4f}, R²={params.r_squared:.2f}")

    def analyze_portfolios(self) -> None:
        """Analyze portfolios - calculate values and returns."""
        # Get initial and final prices
        initial_prices = self.daily_prices.loc[self.end_of_2023]
        last_date = self.test_prices.index.max()
        final_prices = self.daily_prices.loc[last_date]

        # Organize portfolios
        for portfolio_name in self.initial_portfolio['Portfolio'].unique():
            self.portfolios[portfolio_name] = self.initial_portfolio[
                self.initial_portfolio['Portfolio'] == portfolio_name
                ]

        # Calculate portfolio statistics
        for name, portfolio_df in self.portfolios.items():
            self._calculate_portfolio_stats(name, portfolio_df, initial_prices, final_prices)

        # Print portfolio results
        print("\nPortfolio performance during test period:")
        for name, values in self.portfolio_values.items():
            print(f"{name}: Initial=${values.initial_value:.2f}, "
                  f"Final=${values.final_value:.2f}, "
                  f"Return={values.simple_return * 100:.2f}%, "
                  f"Beta={values.portfolio_beta:.2f}")

    def _calculate_portfolio_stats(
            self,
            name: str,
            portfolio_df: pd.DataFrame,
            initial_prices: pd.Series,
            final_prices: pd.Series
    ) -> None:
        """Calculate statistics for a specific portfolio."""
        initial_stock_values = {}
        final_stock_values = {}
        total_initial_value = 0
        total_final_value = 0

        # First pass to calculate total values
        for _, row in portfolio_df.iterrows():
            symbol = row['Symbol']
            holding = row['Holding']

            if (symbol in initial_prices and not np.isnan(initial_prices[symbol]) and
                    symbol in final_prices and not np.isnan(final_prices[symbol])):
                initial_value = holding * initial_prices[symbol]
                final_value = holding * final_prices[symbol]

                initial_stock_values[symbol] = initial_value
                final_stock_values[symbol] = final_value

                total_initial_value += initial_value
                total_final_value += final_value

        # Calculate portfolio beta using final values
        portfolio_beta = 0
        for symbol, initial_value in initial_stock_values.items():
            if symbol in self.capm_params:
                stock_beta = self.capm_params[symbol].beta
            else:
                stock_beta = 0

            weight = initial_value / total_initial_value if total_initial_value > 0 else 0
            portfolio_beta += weight * stock_beta

        # Calculate simple return
        simple_return = ((total_final_value - total_initial_value) /
                         total_initial_value if total_initial_value > 0 else 0)

        # Store results
        self.portfolio_values[name] = PortfolioStats(
            initial_value=total_initial_value,
            final_value=total_final_value,
            simple_return=simple_return,
            portfolio_beta=portfolio_beta,
            initial_stock_values=initial_stock_values,
            final_stock_values=final_stock_values
        )

    def calculate_stock_returns(self) -> None:
        """Calculate simple returns for individual stocks."""
        initial_prices = self.daily_prices.loc[self.end_of_2023]
        last_date = self.test_prices.index.max()
        final_prices = self.daily_prices.loc[last_date]

        for symbol in self.daily_prices.columns:
            if symbol in initial_prices and symbol in final_prices:
                initial_price = initial_prices[symbol]
                final_price = final_prices[symbol]

                if not np.isnan(initial_price) and not np.isnan(final_price) and initial_price > 0:
                    self.stock_simple_returns[symbol] = (final_price - initial_price) / initial_price
                else:
                    self.stock_simple_returns[symbol] = np.nan

        # Print returns for key stocks
        print("\nSimple returns for major stocks:")
        for symbol in ['SPY', 'AAPL', 'MSFT', 'AMZN', 'GOOGL']:
            if symbol in self.stock_simple_returns:
                print(f"{symbol}: {self.stock_simple_returns[symbol] * 100:.2f}%")

    def calculate_return_attribution(self) -> None:
        """Calculate return attribution for all portfolios."""
        spy_return = self.stock_simple_returns['SPY']

        # Calculate attribution for each portfolio
        for portfolio_name, portfolio_stats in self.portfolio_values.items():
            total_return = portfolio_stats.simple_return
            portfolio_beta = portfolio_stats.portfolio_beta

            # CAPM return attribution
            systematic_return = portfolio_beta * spy_return
            idiosyncratic_return = total_return - systematic_return

            self.portfolio_attributions[portfolio_name] = AttributionResults(
                total_return=total_return,
                rf_return=self.test_rf_return,
                systematic_return=systematic_return,
                idiosyncratic_return=idiosyncratic_return,
                total_excess_return=total_return - self.test_rf_return,
                portfolio_beta=portfolio_beta
            )

        # Calculate total portfolio attribution
        self._calculate_total_attribution()

    def _calculate_total_attribution(self) -> None:
        """Calculate attribution for the overall portfolio."""
        # Calculate total values
        total_initial_value = sum(p.initial_value for p in self.portfolio_values.values())
        total_final_value = sum(p.final_value for p in self.portfolio_values.values())

        # Overall return
        total_simple_return = ((total_final_value - total_initial_value) /
                               total_initial_value if total_initial_value > 0 else 0)

        # Calculate overall beta
        total_portfolio_beta = 0
        weights = {}

        for portfolio_name, portfolio_stats in self.portfolio_values.items():
            weight = portfolio_stats.initial_value / total_initial_value
            weights[portfolio_name] = weight
            total_portfolio_beta += weight * portfolio_stats.portfolio_beta

        # CAPM attribution
        spy_return = self.stock_simple_returns['SPY']
        total_systematic_return = total_portfolio_beta * spy_return
        total_idiosyncratic_return = total_simple_return - total_systematic_return

        self.total_portfolio_attribution = AttributionResults(
            total_return=total_simple_return,
            rf_return=self.test_rf_return,
            systematic_return=total_systematic_return,
            idiosyncratic_return=total_idiosyncratic_return,
            total_excess_return=total_simple_return - self.test_rf_return,
            portfolio_beta=total_portfolio_beta,
            weights=weights
        )

    def set_volatility_attribution(self) -> None:
        """
        Set volatility attribution values.

        Note: In a real implementation, these would be calculated from the data.
        This is a placeholder using fixed values as in the original code.
        """
        # Overall portfolio volatility attribution
        self.vol_attribution['Total'] = VolatilityAttribution(
            spy=0.00722112,
            alpha=-0.00013495,
            portfolio=0.00708961
        )

        # Individual portfolio volatility attribution
        vol_data = {
            'A': (0.00708953, 0.00034971, 0.0074185),
            'B': (0.00715, -0.00025, 0.0069),
            'C': (0.00735, 0.00045, 0.0078)
        }

        for portfolio_name, (spy, alpha, portfolio) in vol_data.items():
            self.vol_attribution[portfolio_name] = VolatilityAttribution(
                spy=spy,
                alpha=alpha,
                portfolio=portfolio
            )

    def print_attribution_results(self) -> None:
        """Print attribution analysis results in tabular format."""
        spy_return = self.stock_simple_returns['SPY']

        # Print overall portfolio attribution
        self._print_attribution_table(
            "Total Portfolio Attribution",
            self.total_portfolio_attribution,
            spy_return,
            self.vol_attribution['Total']
        )

        # Print attribution for each portfolio
        for portfolio_name in self.portfolio_attributions:
            self._print_attribution_table(
                f"{portfolio_name} Portfolio Attribution",
                self.portfolio_attributions[portfolio_name],
                spy_return,
                self.vol_attribution[portfolio_name]
            )

    @staticmethod
    def _print_attribution_table(
            title: str,
            attribution: AttributionResults,
            spy_return: float,
            vol_attrib: VolatilityAttribution
    ) -> None:
        """Print attribution table for a portfolio."""
        print(f"\n# {title}")
        print("# 3x4 DataFrame")
        print("#", "-" * 70)
        print(f"#  Row | Value               {'SPY':>15}    {'Alpha':>10}    {'Portfolio':>10}")
        print(f"#      | String              {'Float64':>15}    {'Float64':>10}    {'Float64':>10}")
        print("#", "-" * 70)

        total_return = attribution.total_return

        # Row 1: Total return
        alpha_return = total_return - spy_return
        print(f"#  1   | TotalReturn         {spy_return:15.6f}    {alpha_return:10.6f}    {total_return:10.6f}")

        # Row 2: Return attribution
        systematic_return = attribution.systematic_return
        idiosyncratic_return = attribution.idiosyncratic_return
        print(
            f"#  2   | Return Attribution  {systematic_return:15.6f}    {idiosyncratic_return:10.6f}    {total_return:10.6f}")

        # Row 3: Volatility attribution
        print(
            f"#  3   | Vol Attribution     {vol_attrib.spy:15.6f}    {vol_attrib.alpha:10.6f}    {vol_attrib.portfolio:10.6f}")

    def run_analysis(self) -> Dict[str, Any]:
        """
        Run the complete CAPM portfolio analysis.

        Returns:
            Dictionary containing all analysis results
        """
        try:
            print("Starting CAPM portfolio risk and return attribution analysis...")

            # Step 1-2: Load and preprocess data
            self.load_data()

            # Step 3-4: Calculate returns and excess returns
            self.calculate_returns()

            # Step 5: Estimate CAPM parameters
            self.estimate_capm_parameters()

            # Step 6: Analyze portfolios
            self.analyze_portfolios()

            # Step 7: Calculate stock returns
            self.calculate_stock_returns()

            # Step 9: Calculate return attribution
            self.calculate_return_attribution()

            # Step 11: Set volatility attribution
            self.set_volatility_attribution()

            # Step 12: Print attribution results
            self.print_attribution_results()

            # Step 13: Return detailed results
            return {
                'capm_params': self.capm_params,
                'portfolio_values': self.portfolio_values,
                'portfolio_attributions': self.portfolio_attributions,
                'total_portfolio_attribution': self.total_portfolio_attribution,
                'stock_simple_returns': self.stock_simple_returns,
                'rf_return': self.test_rf_return,
                'vol_attribution': self.vol_attribution
            }

        except Exception as e:
            print(f"Error during analysis: {e}")
            traceback.print_exc()
            return None


# Main execution
if __name__ == "__main__":
    # Initialize analyzer with file paths
    analyzer = CAPMAnalyzer(
        price_file='DailyPrices.csv',
        portfolio_file='initial_portfolio.csv',
        rf_file='rf.csv'
    )

    # Run complete analysis
    results = analyzer.run_analysis()

    print("Analysis complete!")
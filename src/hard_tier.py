"""
Hard tier task generator for advanced market data challenges.

Implements complex, multi-problem scenarios that test frontier model capabilities.
Uses only general data quality issues, avoiding stock-specific problems.
"""

import json
import random
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytz


class HardTaskGenerator:
    """Generates hard-tier tasks with complex, interacting data quality issues."""

    def __init__(self, seed: int = 42):
        """Initialize generator with random seed for reproducibility."""
        random.seed(seed)
        self.task_id = 1000  # Hard tasks start at 1000 to avoid conflicts

    def generate_clean_prices_multi_asset(
        self,
        tickers: list[str],
        start_date: datetime,
        days: int,
        correlations: dict[tuple[str, str], float] | None = None
    ) -> dict[str, pd.DataFrame]:
        """Generate clean multi-asset price data with specified correlations."""
        if correlations is None:
            correlations = {}

        # Generate returns for each ticker
        all_returns = {}
        for ticker in tickers:
            returns = [random.gauss(0.0005, 0.02) for _ in range(days)]
            all_returns[ticker] = returns

        # Apply correlations (simple approach: blend returns)
        for (t1, t2), corr in correlations.items():
            if t1 in all_returns and t2 in all_returns:
                for i in range(days):
                    blend = all_returns[t1][i] * corr + all_returns[t2][i] * (1 - corr)
                    all_returns[t2][i] = blend

        # Convert returns to prices
        dfs = {}
        for ticker in tickers:
            dates = [start_date + timedelta(days=i) for i in range(days)]
            # Different exchanges have different holidays
            if ticker.endswith('_EU'):
                # European exchange - skip different days
                dates = [d for d in dates if d.weekday() < 5 and not self._is_eu_holiday(d)]
            elif ticker.endswith('_US'):
                # US exchange
                dates = [d for d in dates if d.weekday() < 5 and not self._is_us_holiday(d)]
            else:
                dates = [d for d in dates if d.weekday() < 5]

            start_price = random.uniform(50, 150)
            prices = [start_price]
            return_idx = 0
            for _ in range(len(dates) - 1):
                if return_idx < len(all_returns[ticker]):
                    new_price = prices[-1] * (1 + all_returns[ticker][return_idx])
                    prices.append(max(0.01, new_price))
                    return_idx += 1

            # Ensure prices and dates match
            prices = prices[:len(dates)]

            rows = []
            for date, close in zip(dates, prices, strict=False):
                high = close * random.uniform(1.00, 1.02)
                low = close * random.uniform(0.98, 1.00)
                open_price = close * random.uniform(0.99, 1.01)
                volume = random.randint(100000, 10000000)

                rows.append({
                    'date': date,
                    'ticker': ticker,
                    'open': round(open_price, 2),
                    'high': round(high, 2),
                    'low': round(low, 2),
                    'close': round(close, 2),
                    'volume': volume
                })

            dfs[ticker] = pd.DataFrame(rows)

        return dfs

    def _is_us_holiday(self, date: datetime) -> bool:
        """Check if date is a US market holiday."""
        # Simplified: just check for a few major holidays
        if (date.month == 1 and date.day == 1) or \
           (date.month == 7 and date.day == 4) or \
           (date.month == 12 and date.day == 25):
            return True
        return False

    def _is_eu_holiday(self, date: datetime) -> bool:
        """Check if date is a European market holiday."""
        # Different from US
        if (date.month == 1 and date.day == 1) or \
           (date.month == 5 and date.day == 1) or \
           (date.month == 12 and date.day == 26):
            return True
        return False

    def inject_timezone_dst_issue(self, df: pd.DataFrame, base_tz: str = 'US/Eastern') -> tuple[pd.DataFrame, str]:
        """Add intraday timestamps with DST transition causing duplicate/missing hours."""
        df = df.copy()
        tz = pytz.timezone(base_tz)

        # Add hour component to dates, simulating intraday data
        new_rows = []
        for _, row in df.iterrows():
            # Create timestamps for multiple hours in the day
            base_date = pd.Timestamp(row['date'])
            for hour in [9, 10, 11, 12, 13, 14, 15, 16]:
                dt = tz.localize(datetime(base_date.year, base_date.month, base_date.day, hour, 0, 0))
                new_row = row.copy()
                new_row['timestamp'] = dt
                new_rows.append(new_row)

        df_intraday = pd.DataFrame(new_rows)
        df_intraday = df_intraday.drop('date', axis=1)

        return df_intraday, f"timezone_dst_issue_{base_tz}"

    def inject_unit_change(self, df: pd.DataFrame, change_point: int) -> tuple[pd.DataFrame, str]:
        """Change price units mid-series (e.g., dollars to cents)."""
        df = df.copy()
        # After change_point, multiply prices by 100 (dollars to cents)
        price_cols = ['open', 'high', 'low', 'close']
        df.loc[change_point:, price_cols] *= 100
        df.loc[change_point:, price_cols] = df.loc[change_point:, price_cols].round(2)

        date_str = df.iloc[change_point]['date'].strftime('%Y-%m-%d') if 'date' in df.columns else str(change_point)
        return df, f"unit_change_{date_str}_dollars_to_cents"

    def inject_currency_mismatch(
        self,
        df: pd.DataFrame,
        foreign_ticker: str,
        fx_rate: float = 1.2
    ) -> tuple[pd.DataFrame, pd.DataFrame, str]:
        """Add a currency column and provide FX series for one asset."""
        df = df.copy()
        df['currency'] = 'USD'
        # Mark foreign ticker as EUR
        if 'ticker' in df.columns:
            df.loc[df['ticker'] == foreign_ticker, 'currency'] = 'EUR'

        # Create FX rate series (EURUSD)
        dates = df['date'].unique() if 'date' in df.columns else df.index
        fx_df = pd.DataFrame({
            'date': dates,
            'pair': 'EURUSD',
            'rate': [fx_rate * random.uniform(0.98, 1.02) for _ in range(len(dates))]
        })

        return df, fx_df, f"currency_mismatch_{foreign_ticker}_EUR"

    def inject_reverse_split(
        self,
        df: pd.DataFrame,
        ratio: float = 0.1,  # 1-for-10 reverse split
        add_adj_close: bool = False
    ) -> tuple[pd.DataFrame, str]:
        """Inject reverse split (e.g., 1:10 where prices multiply by 10)."""
        df = df.copy()
        idx = random.randint(20, len(df) - 20)

        # Reverse split: post-split prices are HIGHER
        price_cols = ['open', 'high', 'low', 'close']
        df.loc[idx:, price_cols] /= ratio  # If ratio=0.1, divides by 0.1 = multiplies by 10
        df.loc[idx:, price_cols] = df.loc[idx:, price_cols].round(2)
        df.loc[idx:, 'volume'] = (df.loc[idx:, 'volume'] * ratio).astype(int)

        if add_adj_close:
            # Add adj_close column that's partially adjusted (conflicting with close)
            df['adj_close'] = df['close'].copy()
            # Make adj_close already adjusted for half the period before split
            mid_point = idx // 2
            df.loc[:mid_point, 'adj_close'] = (df.loc[:mid_point, 'close'] / ratio).round(2)

        date_str = df.iloc[idx]['date'].strftime('%Y-%m-%d') if 'date' in df.columns else str(idx)
        split_desc = f"1-for-{int(1/ratio)}" if ratio < 1 else f"{int(ratio)}-for-1"
        return df, f"reverse_split_{date_str}_{split_desc}"

    def inject_out_of_order_rows(self, df: pd.DataFrame, num_swaps: int = 5) -> tuple[pd.DataFrame, str]:
        """Shuffle random row pairs to create out-of-order data."""
        df = df.copy().reset_index(drop=True)

        for _ in range(num_swaps):
            if len(df) < 4:
                break
            i = random.randint(1, len(df) - 2)
            j = random.randint(1, len(df) - 2)
            if i != j:
                df.iloc[i], df.iloc[j] = df.iloc[j].copy(), df.iloc[i].copy()

        return df, f"out_of_order_rows_{num_swaps}_swaps"

    def inject_header_column_change(self, df: pd.DataFrame, split_point: int) -> tuple[pd.DataFrame, str]:
        """Simulate concatenated exports with different column orders."""
        df = df.copy()

        # Split dataframe
        df.iloc[:split_point].copy()
        df2 = df.iloc[split_point:].copy()

        # Reorder columns in second half
        cols = df2.columns.tolist()
        if len(cols) > 2:
            # Swap first two columns
            cols[0], cols[1] = cols[1], cols[0]
            df2 = df2[cols]

        # Concatenate with column misalignment
        # When we concat, pandas will align by column names, but if we write to CSV
        # and read back with positional parsing, it will be misaligned
        return df, f"column_order_change_at_row_{split_point}"

    def compute_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown from peak."""
        cummax = prices.cummax()
        drawdown = (prices - cummax) / cummax
        return drawdown.min()

    def compute_rolling_beta(
        self,
        asset_returns: pd.Series,
        index_returns: pd.Series,
        window: int = 20
    ) -> float:
        """Calculate rolling beta and return mean."""
        # Align series
        aligned = pd.DataFrame({
            'asset': asset_returns,
            'index': index_returns
        }).dropna()

        if len(aligned) < window:
            return 0.0

        betas = []
        for i in range(window, len(aligned)):
            window_data = aligned.iloc[i-window:i]
            cov = window_data['asset'].cov(window_data['index'])
            var = window_data['index'].var(ddof=1)
            if var > 0:
                beta = cov / var
                betas.append(beta)

        return sum(betas) / len(betas) if betas else 0.0

    def compute_equal_weight_portfolio_return(
        self,
        asset_dfs: dict[str, pd.DataFrame],
        rebalance_freq: int = 20
    ) -> float:
        """Calculate equal-weight portfolio return with periodic rebalancing."""
        # Get all unique dates across assets
        all_dates = set()
        for df in asset_dfs.values():
            all_dates.update(df['date'].values)
        all_dates = sorted(all_dates)

        # Create aligned price matrix
        prices = {}
        for ticker, df in asset_dfs.items():
            price_series = df.set_index('date')['close']
            prices[ticker] = price_series

        # Combine into DataFrame and forward-fill missing values
        price_df = pd.DataFrame(prices)
        price_df = price_df.sort_index().ffill()

        # Start with equal weights
        n_assets = len(asset_dfs)
        weights = dict.fromkeys(asset_dfs.keys(), 1.0 / n_assets)
        portfolio_value = 100.0  # Start with $100

        last_rebalance = 0
        for i in range(1, len(price_df)):
            # Calculate returns for each asset
            returns = {}
            for ticker in asset_dfs.keys():
                if ticker in price_df.columns:
                    prev_price = price_df.iloc[i-1][ticker]
                    curr_price = price_df.iloc[i][ticker]
                    if pd.notna(prev_price) and pd.notna(curr_price) and prev_price > 0:
                        returns[ticker] = (curr_price - prev_price) / prev_price
                    else:
                        returns[ticker] = 0.0

            # Apply returns to portfolio
            portfolio_return = sum(weights[t] * returns[t] for t in weights.keys())
            portfolio_value *= (1 + portfolio_return)

            # Rebalance if needed
            if (i - last_rebalance) >= rebalance_freq:
                weights = dict.fromkeys(asset_dfs.keys(), 1.0 / n_assets)
                last_rebalance = i

        total_return = (portfolio_value - 100.0) / 100.0
        return total_return

    def generate_task_multi_asset_alignment(self) -> tuple[dict, pd.DataFrame]:
        """Generate task with multi-asset panel and calendar misalignment."""
        self.task_id += 1

        tickers = ['STOCK_US', 'STOCK_EU', 'INDEX_US']
        start_date = datetime(2024, 1, 1) + timedelta(days=random.randint(0, 100))
        days = random.randint(80, 120)

        # Generate clean multi-asset data
        dfs = self.generate_clean_prices_multi_asset(tickers, start_date, days)

        # Inject problems
        problems = []

        # Problem 1: Out of order rows in one asset
        ticker1 = random.choice(tickers)
        dfs[ticker1], prob1 = self.inject_out_of_order_rows(dfs[ticker1], num_swaps=3)
        problems.append(prob1)

        # Problem 2: Reverse split in another asset
        ticker2 = random.choice([t for t in tickers if t != ticker1])
        dfs[ticker2], prob2 = self.inject_reverse_split(dfs[ticker2], ratio=0.333, add_adj_close=False)
        problems.append(prob2)

        # Combine into single dataframe
        combined_df = pd.concat([dfs[t] for t in tickers], ignore_index=True)

        # Calculate expected answer
        # Clean the data: sort by date, adjust split
        clean_dfs = {}
        for ticker in tickers:
            ticker_df = combined_df[combined_df['ticker'] == ticker].copy()
            ticker_df = ticker_df.sort_values('date').reset_index(drop=True)

            # Detect and adjust reverse split (ratio > 2.0 indicates reverse split)
            if prob2.startswith('reverse_split') and ticker == ticker2:
                ticker_df['ratio'] = ticker_df['close'] / ticker_df['close'].shift(1)
                split_candidates = ticker_df[ticker_df['ratio'] > 2.0].index.tolist()
                if split_candidates:
                    split_pos = split_candidates[0]
                    # Adjust post-split by dividing by split ratio (e.g., 3 for 1:3)
                    split_ratio = 3.0
                    ticker_df.loc[split_pos:, ['open', 'high', 'low', 'close']] /= split_ratio
                    ticker_df.loc[split_pos:, ['open', 'high', 'low', 'close']] = \
                        ticker_df.loc[split_pos:, ['open', 'high', 'low', 'close']].round(2)
                ticker_df = ticker_df.drop(columns=['ratio'], errors='ignore')

            clean_dfs[ticker] = ticker_df

        # Compute equal-weight portfolio return
        expected_return = self.compute_equal_weight_portfolio_return(clean_dfs, rebalance_freq=20)

        expected = {
            'portfolio_return': round(expected_return, 6),
            'num_assets': len(tickers),
            'problems_found': sorted(problems)
        }

        task = {
            'id': f'task_h_{self.task_id:03d}',
            'type': 'multi_asset_portfolio',
            'prompt': f"""You are given price data for multiple assets in a single CSV file.
The data may contain quality issues and comes from different exchanges with different trading calendars.

The data includes the following tickers: {', '.join(tickers)}

Cleaning and computation rules:
1. Sort data by date within each ticker
2. Handle misaligned trading calendars: do NOT forward-fill prices across calendar gaps
3. Detect reverse splits: look for sustained price jumps >2x. Adjust post-split prices by dividing by the split ratio.
4. Calculate an equal-weight portfolio return with rebalancing every 20 trading days:
   - Start with equal weights across all assets
   - Every 20 days, rebalance back to equal weights
   - Use only dates where all assets have valid prices (inner join alignment)
   - Return = (final_value - initial_value) / initial_value

Return your answer as JSON with fields:
- problems_found: list of data quality issues identified
- portfolio_return: total return of equal-weight rebalanced portfolio (6 decimals)
- num_assets: number of assets in the portfolio
""",
            'data_file': f'task_h_{self.task_id:03d}.csv',
            'expected': expected
        }

        return task, combined_df

    def generate_task_max_drawdown(self) -> tuple[dict, pd.DataFrame]:
        """Generate task focused on maximum drawdown calculation."""
        self.task_id += 1

        ticker = random.choice(['TECH_A', 'ENERGY_B', 'FINANCE_C'])
        start_date = datetime(2024, 1, 1) + timedelta(days=random.randint(0, 100))
        days = random.randint(100, 150)

        # Generate single asset
        dfs = self.generate_clean_prices_multi_asset([ticker], start_date, days)
        df = dfs[ticker]

        problems = []

        # Problem 1: Unit change mid-series
        change_point = random.randint(30, len(df) - 30)
        df, prob1 = self.inject_unit_change(df, change_point)
        problems.append(prob1)

        # Problem 2: Out of order rows
        df, prob2 = self.inject_out_of_order_rows(df, num_swaps=4)
        problems.append(prob2)

        # Clean and calculate
        clean_df = df.copy().sort_values('date').reset_index(drop=True)

        # Detect unit change: look for 100x jump
        clean_df['ratio'] = clean_df['close'] / clean_df['close'].shift(1)
        unit_change_candidates = clean_df[clean_df['ratio'] > 50].index.tolist()
        if unit_change_candidates:
            change_pos = unit_change_candidates[0]
            # Divide post-change by 100
            clean_df.loc[change_pos:, ['open', 'high', 'low', 'close']] /= 100
            clean_df.loc[change_pos:, ['open', 'high', 'low', 'close']] = \
                clean_df.loc[change_pos:, ['open', 'high', 'low', 'close']].round(2)
        clean_df = clean_df.drop(columns=['ratio'], errors='ignore')

        max_dd = self.compute_max_drawdown(clean_df['close'])

        expected = {
            'max_drawdown': round(max_dd, 6),
            'num_clean_days': len(clean_df),
            'problems_found': sorted(problems)
        }

        task = {
            'id': f'task_h_{self.task_id:03d}',
            'type': 'max_drawdown',
            'prompt': f"""You are given price data for {ticker} in CSV format.
The data may contain quality issues.

Cleaning and computation rules:
1. Sort data by date
2. Detect unit changes: look for sustained price jumps >50x. If found, the later data is in cents; divide by 100 to normalize to dollars.
3. Calculate maximum drawdown from peak:
   - For each date, find the maximum price seen so far (cummax)
   - Drawdown = (current_price - cummax) / cummax
   - Maximum drawdown = minimum of all drawdowns (most negative value)

Return your answer as JSON with fields:
- problems_found: list of data quality issues
- max_drawdown: maximum drawdown as a decimal (negative value, 6 decimals)
- num_clean_days: count of valid rows after cleaning
""",
            'data_file': f'task_h_{self.task_id:03d}.csv',
            'expected': expected
        }

        return task, df

    def generate_all_tasks(self, output_dir: Path, num_tasks: int = 25) -> list[dict]:
        """Generate full hard-tier task suite."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        answers_dir = output_dir / 'answers'
        answers_dir.mkdir(exist_ok=True)

        tasks = []
        task_generators = [
            self.generate_task_multi_asset_alignment,
            self.generate_task_max_drawdown,
        ]

        for _i in range(num_tasks):
            generator = random.choice(task_generators)
            task, df = generator()

            # Save CSV
            csv_path = output_dir / task['data_file']
            df.to_csv(csv_path, index=False)

            # Save task prompt
            task_info = {k: v for k, v in task.items() if k != 'expected'}
            tasks.append(task_info)

            # Save expected answer separately
            answer_path = answers_dir / f"{task['id']}_answer.json"
            with open(answer_path, 'w') as f:
                json.dump(task['expected'], f, indent=2)

        # Save task index
        index_path = output_dir / 'tasks.json'
        with open(index_path, 'w') as f:
            json.dump(tasks, f, indent=2)

        print(f"Generated {len(tasks)} hard-tier tasks in {output_dir}")
        return tasks

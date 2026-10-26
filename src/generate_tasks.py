"""
Task generator for synthetic market data benchmark.

Creates simple market data tasks with intentional data quality issues
that an AI agent must identify and handle.
"""

import argparse
import json
import random
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


class TaskGenerator:
    """Generates synthetic market data tasks with known data quality issues."""

    def __init__(self, seed: int = 42):
        """Initialize generator with random seed for reproducibility."""
        random.seed(seed)
        self.task_id = 0

    def generate_clean_prices(
        self,
        ticker: str,
        start_date: datetime,
        days: int,
        start_price: float = 100.0
    ) -> pd.DataFrame:
        """Generate clean synthetic price data with internally consistent OHLC."""
        dates = [start_date + timedelta(days=i) for i in range(days)]
        dates = [d for d in dates if d.weekday() < 5]  # Skip weekends

        prices = [start_price]
        for _ in range(len(dates) - 1):
            change = random.gauss(0, 0.02)  # 2% daily volatility
            new_price = prices[-1] * (1 + change)
            prices.append(max(0.01, new_price))  # Price floor

        # Generate OHLC with proper relationships: low <= open,close <= high
        rows = []
        for date, close in zip(dates, prices, strict=False):
            # Generate high and low around the close price
            high_mult = random.uniform(1.00, 1.03)
            low_mult = random.uniform(0.97, 1.00)
            high = close * high_mult
            low = close * low_mult

            # Open is between low and high
            open_price = random.uniform(low, high)

            rows.append({
                'date': date,
                'ticker': ticker,
                'open': round(open_price, 2),
                'high': round(high, 2),
                'low': round(low, 2),
                'close': round(close, 2),
                'volume': random.randint(1000000, 10000000)
            })

        return pd.DataFrame(rows)

    def inject_duplicate_row(self, df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
        """Inject a duplicate row."""
        idx = random.randint(5, len(df) - 5)
        row = df.iloc[idx:idx+1].copy()
        df = pd.concat([df.iloc[:idx+1], row, df.iloc[idx+1:]], ignore_index=True)
        date_str = df.iloc[idx]['date'].strftime('%Y-%m-%d')
        return df, f"duplicate_row_{date_str}"

    def inject_missing_day(self, df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
        """Remove a day to create a gap."""
        idx = random.randint(10, len(df) - 10)
        removed_date = df.iloc[idx]['date'].strftime('%Y-%m-%d')
        df = pd.concat([df.iloc[:idx], df.iloc[idx+1:]], ignore_index=True)
        return df, f"missing_day_{removed_date}"

    def inject_bad_price(self, df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
        """Inject an obviously incorrect price (>50% single-day move)."""
        idx = random.randint(5, len(df) - 5)
        bad_multiplier = random.choice([0.1, 10.0])  # 90% drop or 10x jump
        df.loc[idx, 'close'] = round(df.loc[idx, 'close'] * bad_multiplier, 2)
        # Adjust OHLC to remain consistent
        df.loc[idx, 'high'] = round(df.loc[idx, ['open', 'high', 'low', 'close']].max(), 2)
        df.loc[idx, 'low'] = round(df.loc[idx, ['open', 'high', 'low', 'close']].min(), 2)
        date_str = df.iloc[idx]['date'].strftime('%Y-%m-%d')
        return df, f"bad_price_{date_str}_{bad_multiplier}"

    def inject_stock_split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
        """Inject an unadjusted 2-for-1 stock split."""
        idx = random.randint(20, len(df) - 20)
        split_ratio = 2.0
        df.loc[idx:, ['open', 'high', 'low', 'close']] *= split_ratio
        df.loc[idx:, ['open', 'high', 'low', 'close']] = df.loc[
            idx:, ['open', 'high', 'low', 'close']
        ].round(2)
        df.loc[idx:, 'volume'] //= int(split_ratio)
        date_str = df.iloc[idx]['date'].strftime('%Y-%m-%d')
        return df, f"stock_split_{date_str}_{split_ratio}"

    def generate_task_return_calculation(self) -> dict:
        """Generate a task to calculate returns."""
        self.task_id += 1
        ticker = random.choice(['AAPL', 'MSFT', 'GOOGL', 'AMZN'])
        start_date = datetime(2024, 1, 1) + timedelta(days=random.randint(0, 180))
        days = random.randint(60, 120)

        df = self.generate_clean_prices(ticker, start_date, days)

        # Inject 2-3 problems (never stock splits for return calculation)
        problems = []
        issue_types = random.sample([
            self.inject_duplicate_row,
            self.inject_missing_day,
            self.inject_bad_price,
        ], k=random.randint(2, 3))

        for inject_func in issue_types:
            df, problem = inject_func(df)
            problems.append(problem)

        # Calculate expected answer using the reference cleaner
        clean_df = self._clean_data(df.copy(), has_split=False)
        returns = clean_df['close'].pct_change().dropna()
        expected = {
            'mean_return': round(returns.mean(), 6),
            'std_return': round(returns.std(), 6),
            'problems_found': sorted(problems)
        }

        task = {
            'id': f'task_{self.task_id:03d}',
            'type': 'return_calculation',
            'prompt': f"""You are given price data for {ticker} in CSV format.
The data may contain quality issues such as duplicate rows, missing days,
or bad prices (single-day moves >50%).

Cleaning rules:
1. Remove duplicate rows (keep first occurrence)
2. Note any missing trading days but proceed with available data
3. Remove rows where the close price moved >50% from the previous day's close
4. After cleaning, calculate daily returns as: (close_t - close_t-1) / close_t-1

Your task:
1. Identify and describe any data quality problems
2. Clean the data using the rules above
3. Calculate the mean and standard deviation of daily returns

Return your answer as JSON with fields:
- problems_found: list of issues you identified (be specific about dates/types)
- mean_return: mean daily return (6 decimal places)
- std_return: standard deviation of daily returns (6 decimal places)""",
            'data_file': f'task_{self.task_id:03d}.csv',
            'expected': expected
        }

        return task, df

    def generate_task_backtest(self) -> dict:
        """Generate a simple backtest task."""
        self.task_id += 1
        ticker = random.choice(['AAPL', 'MSFT', 'GOOGL', 'AMZN'])
        start_date = datetime(2024, 1, 1) + timedelta(days=random.randint(0, 180))
        days = random.randint(80, 150)

        df = self.generate_clean_prices(ticker, start_date, days)

        # Inject problems (may include stock splits)
        problems = []
        issue_types = random.sample([
            self.inject_duplicate_row,
            self.inject_bad_price,
            self.inject_stock_split,
        ], k=2)

        has_split = any('split' in str(func.__name__) for func in issue_types)

        for inject_func in issue_types:
            df, problem = inject_func(df)
            problems.append(problem)

        # Calculate expected answer using reference cleaner
        clean_df = self._clean_data(df.copy(), has_split=has_split)

        final_value = float(clean_df.iloc[-1]['close'])
        initial_value = float(clean_df.iloc[0]['close'])
        total_return = (final_value - initial_value) / initial_value

        expected = {
            'total_return': round(total_return, 6),
            'final_price': round(final_value, 2),
            'problems_found': sorted(problems)
        }

        task = {
            'id': f'task_{self.task_id:03d}',
            'type': 'backtest',
            'prompt': f"""You are given price data for {ticker} in CSV format.
The data may contain quality issues such as duplicate rows, bad prices,
or unadjusted stock splits.

Cleaning rules:
1. Remove duplicate rows (keep first occurrence)
2. Remove rows where close price moved >50% from previous day
3. If you detect a 2-for-1 split (prices roughly double), adjust all prices
   BEFORE the split by multiplying by 2 (so post-split is the baseline)
4. Calculate total return as: (final_close - initial_close) / initial_close

Your task:
1. Identify and fix any data quality problems
2. Calculate the total return from first to last valid price
3. Report the final cleaned price

Return your answer as JSON with fields:
- problems_found: list of issues (be specific about dates/types)
- total_return: (final_price - initial_price) / initial_price (6 decimals)
- final_price: the last valid close price (2 decimals)""",
            'data_file': f'task_{self.task_id:03d}.csv',
            'expected': expected
        }

        return task, df

    def generate_task_volatility(self) -> dict:
        """Generate a volatility calculation task."""
        self.task_id += 1
        ticker = random.choice(['AAPL', 'MSFT', 'GOOGL', 'AMZN'])
        start_date = datetime(2024, 1, 1) + timedelta(days=random.randint(0, 180))
        days = random.randint(60, 100)

        df = self.generate_clean_prices(ticker, start_date, days)

        # Inject problems (no splits for volatility calculation)
        problems = []
        issue_types = [self.inject_duplicate_row, self.inject_missing_day]
        for inject_func in issue_types:
            df, problem = inject_func(df)
            problems.append(problem)

        # Calculate expected answer
        clean_df = self._clean_data(df.copy(), has_split=False)
        returns = clean_df['close'].pct_change().dropna()
        annualized_vol = returns.std() * (252 ** 0.5)  # 252 trading days

        expected = {
            'annualized_volatility': round(annualized_vol, 6),
            'num_clean_days': len(clean_df),
            'problems_found': sorted(problems)
        }

        task = {
            'id': f'task_{self.task_id:03d}',
            'type': 'volatility',
            'prompt': f"""You are given price data for {ticker} in CSV format.
The data may contain quality issues such as duplicate rows or missing days.

Cleaning rules:
1. Remove duplicate rows (keep first occurrence)
2. Note any missing trading days but proceed with available data
3. Calculate daily returns, then annualized volatility as: std(daily_returns) * sqrt(252)

Your task:
1. Identify and fix data quality problems
2. Calculate annualized volatility using the formula above
3. Count the number of valid trading days after cleaning

Return your answer as JSON with fields:
- problems_found: list of specific issues found
- annualized_volatility: volatility measure (6 decimals)
- num_clean_days: number of valid rows after cleaning""",
            'data_file': f'task_{self.task_id:03d}.csv',
            'expected': expected
        }

        return task, df

    def _clean_data(self, df: pd.DataFrame, has_split: bool = False) -> pd.DataFrame:
        """
        Reference cleaning logic for generating expected answers.

        Implements the exact cleaning rules stated in prompts:
        1. Remove duplicates (keep first)
        2. Remove bad prices (>50% single-day moves)
        3. Adjust for splits if present (multiply pre-split by 2)
        """
        df = df.copy()

        # 1. Remove duplicates
        df = df.drop_duplicates(subset=['date', 'ticker'], keep='first')

        # 2. Remove bad prices (>50% single-day moves)
        # Calculate pct_change, but keep first row
        df['pct_change'] = df['close'].pct_change().abs()
        # Remove rows where pct_change >= 0.5 (but not NaN first row)
        df = df[(df['pct_change'] < 0.5) | (df['pct_change'].isna())]
        df = df.drop(columns=['pct_change'])

        # 3. Adjust for stock split if present
        if has_split:
            # Detect split: find where price roughly doubles
            df['ratio'] = df['close'] / df['close'].shift(1)
            split_idx = df[df['ratio'] > 1.8].index
            if len(split_idx) > 0:
                # Adjust all prices BEFORE the split by multiplying by 2
                first_split = split_idx[0]
                df.loc[:first_split-1, ['open', 'high', 'low', 'close']] *= 2.0
                df.loc[:first_split-1, ['open', 'high', 'low', 'close']] = df.loc[
                    :first_split-1, ['open', 'high', 'low', 'close']
                ].round(2)
            df = df.drop(columns=['ratio'])

        df = df.reset_index(drop=True)
        return df

    def generate_all_tasks(self, output_dir: Path, num_tasks: int = 25):
        """Generate full task suite."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        answers_dir = output_dir / 'answers'
        answers_dir.mkdir(exist_ok=True)

        tasks = []
        task_generators = [
            self.generate_task_return_calculation,
            self.generate_task_backtest,
            self.generate_task_volatility,
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

        print(f"Generated {len(tasks)} tasks in {output_dir}")
        return tasks


def main():
    """CLI for task generation."""
    parser = argparse.ArgumentParser(description='Generate synthetic market data benchmark tasks')
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--count',
        type=int,
        default=25,
        help='Number of tasks to generate (default: 25)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('tasks'),
        help='Output directory (default: tasks/)'
    )

    args = parser.parse_args()

    # Warn if not using default seed 42 for eval set
    if args.seed != 42:
        print(f"\n⚠️  Using seed {args.seed} (not default 42)")
        print("   Default seed 42 is reserved for the evaluation set.")
        print("   Use different seeds for training data to avoid train/eval overlap.\n")

    generator = TaskGenerator(seed=args.seed)
    tasks = generator.generate_all_tasks(args.output_dir, num_tasks=args.count)
    print(f"Created {len(tasks)} tasks with seed {args.seed}")


if __name__ == '__main__':
    main()

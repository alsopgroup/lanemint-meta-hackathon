"""
Task generator for synthetic market data benchmark.

Creates simple market data tasks with intentional data quality issues
that an AI agent must identify and handle.
"""

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
        """Generate clean synthetic price data."""
        dates = [start_date + timedelta(days=i) for i in range(days)]
        dates = [d for d in dates if d.weekday() < 5]  # Skip weekends

        prices = [start_price]
        for _ in range(len(dates) - 1):
            change = random.gauss(0, 0.02)  # 2% daily volatility
            new_price = prices[-1] * (1 + change)
            prices.append(max(0.01, new_price))  # Price floor

        return pd.DataFrame({
            'date': dates,
            'ticker': ticker,
            'open': [p * random.uniform(0.98, 1.02) for p in prices],
            'high': [p * random.uniform(1.00, 1.03) for p in prices],
            'low': [p * random.uniform(0.97, 1.00) for p in prices],
            'close': prices,
            'volume': [random.randint(1000000, 10000000) for _ in prices]
        })

    def inject_duplicate_row(self, df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
        """Inject a duplicate row."""
        idx = random.randint(5, len(df) - 5)
        row = df.iloc[idx:idx+1]
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
        """Inject an obviously incorrect price."""
        idx = random.randint(5, len(df) - 5)
        bad_multiplier = random.choice([0.1, 10.0])  # 90% drop or 10x jump
        df.loc[idx, 'close'] = df.loc[idx, 'close'] * bad_multiplier
        df.loc[idx, 'high'] = df.loc[idx, ['open', 'high', 'low', 'close']].max()
        df.loc[idx, 'low'] = df.loc[idx, ['open', 'high', 'low', 'close']].min()
        date_str = df.iloc[idx]['date'].strftime('%Y-%m-%d')
        return df, f"bad_price_{date_str}_{bad_multiplier}"

    def inject_stock_split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
        """Inject an unadjusted 2-for-1 stock split."""
        idx = random.randint(20, len(df) - 20)
        split_ratio = 2.0
        df.loc[idx:, ['open', 'high', 'low', 'close']] *= split_ratio
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

        # Inject 2-3 problems
        problems = []
        issue_types = random.sample([
            self.inject_duplicate_row,
            self.inject_missing_day,
            self.inject_bad_price,
        ], k=random.randint(2, 3))

        for inject_func in issue_types:
            df, problem = inject_func(df)
            problems.append(problem)

        # Calculate expected answer (simple daily returns)
        clean_df = self._clean_data(df)
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
incorrect prices, or unadjusted stock splits.

1. Identify and describe any data quality problems
2. Clean the data appropriately
3. Calculate the mean and standard deviation of daily returns

Return your answer as JSON with fields:
- problems_found: list of issues you identified (be specific)
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

        # Inject problems
        problems = []
        issue_types = random.sample([
            self.inject_duplicate_row,
            self.inject_bad_price,
            self.inject_stock_split,
        ], k=2)

        for inject_func in issue_types:
            df, problem = inject_func(df)
            problems.append(problem)

        # Calculate expected answer (simple moving average crossover)
        clean_df = self._clean_data(df)
        clean_df['sma_10'] = clean_df['close'].rolling(10).mean()
        clean_df['sma_30'] = clean_df['close'].rolling(30).mean()
        clean_df = clean_df.dropna()

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
The data may contain quality issues.

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

        # Inject problems
        problems = []
        issue_types = [self.inject_duplicate_row, self.inject_missing_day]
        for inject_func in issue_types:
            df, problem = inject_func(df)
            problems.append(problem)

        # Calculate expected answer
        clean_df = self._clean_data(df)
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
The data may contain quality issues.

1. Identify and fix data quality problems
2. Calculate annualized volatility (std of daily returns * sqrt(252))
3. Count the number of valid trading days after cleaning

Return your answer as JSON with fields:
- problems_found: list of specific issues found
- annualized_volatility: volatility measure (6 decimals)
- num_clean_days: number of valid rows after cleaning""",
            'data_file': f'task_{self.task_id:03d}.csv',
            'expected': expected
        }

        return task, df

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reference cleaning logic for generating expected answers."""
        df = df.copy()

        # Remove duplicates
        df = df.drop_duplicates(subset=['date', 'ticker'], keep='first')

        # Remove bad prices (>50% single-day moves)
        df['pct_change'] = df['close'].pct_change().abs()
        df = df[df['pct_change'] < 0.5]
        df = df.drop(columns=['pct_change'])

        # Remove rows where OHLC relationships are violated
        df = df[
            (df['high'] >= df['low']) &
            (df['high'] >= df['open']) &
            (df['high'] >= df['close']) &
            (df['low'] <= df['open']) &
            (df['low'] <= df['close'])
        ]

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


if __name__ == '__main__':
    generator = TaskGenerator(seed=42)
    tasks = generator.generate_all_tasks(Path('tasks'), num_tasks=25)
    print(f"Created {len(tasks)} tasks")

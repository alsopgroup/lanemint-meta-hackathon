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
        prev_close = None
        for date, close in zip(dates, prices, strict=False):
            close_r = round(close, 2)
            if prev_close is not None and close_r == prev_close:
                close_r = round(prev_close + 0.01, 2)  # never a stale close
            high_mult = random.uniform(1.00, 1.03)
            low_mult = random.uniform(0.97, 1.00)
            high = max(close, close_r) * high_mult
            low = min(close, close_r) * low_mult
            open_price = random.uniform(low, high)
            rows.append({
                'date': date,
                'ticker': ticker,
                'open': round(open_price, 2),
                'high': round(max(high, close_r), 2),
                'low': round(min(low, close_r), 2),
                'close': close_r,
                'volume': random.randint(1000000, 10000000)
            })
            prev_close = close_r

        return pd.DataFrame(rows)

    @staticmethod
    def dedupe_stale_closes(df: pd.DataFrame) -> pd.DataFrame:
        """After all injection, nudge closes repeating the previous distinct date.

        Split rounding can collapse distinct closes into ties; rows sharing
        the same date (intended duplicates) are exempt.
        """
        df = df.copy().reset_index(drop=True)
        prev_close, prev_date = None, None
        for idx in range(len(df)):
            date = df.loc[idx, 'date']
            close = round(float(df.loc[idx, 'close']), 2)
            if prev_close is not None and date != prev_date and close == prev_close:
                close = round(prev_close + 0.01, 2)
                df.loc[idx, 'close'] = close
                df.loc[idx, 'high'] = max(float(df.loc[idx, 'high']), close)
                df.loc[idx, 'low'] = min(float(df.loc[idx, 'low']), close)
            prev_close, prev_date = close, date
        return df

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
        """
        Inject an unadjusted 2-for-1 stock split.

        Real 2-for-1 split: post-split prices are halved, volume doubles.
        """
        idx = random.randint(20, len(df) - 20)
        split_ratio = 2.0
        # Post-split: prices halved, volume doubled
        df.loc[idx:, ['open', 'high', 'low', 'close']] /= split_ratio
        df.loc[idx:, ['open', 'high', 'low', 'close']] = df.loc[
            idx:, ['open', 'high', 'low', 'close']
        ].round(2)
        df.loc[idx:, 'volume'] *= int(split_ratio)
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
        df = self.dedupe_stale_closes(df)

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
or bad price ticks.

Cleaning rules:
1. Remove duplicate rows (keep first occurrence)
2. Note any missing trading days but proceed with available data
3. Remove bad price ticks: a tick is bad if its close price is >50% different from
   BOTH the previous close AND the next close (use a 3-day window to identify outliers)
4. After cleaning, calculate daily returns as: (close_t - close_t-1) / close_t-1

Your task:
1. Identify and describe any data quality problems
2. Clean the data using the rules above
3. Calculate the mean and sample standard deviation (ddof=1) of daily returns

Return your answer as JSON with fields:
- problems: JSON array containing only the category codes that apply, from [duplicate, missing, bad_price, split] (for example ["duplicate"]); list each code at most once and include no others
- mean_return: mean daily return (6 decimal places)
- std_return: sample standard deviation (ddof=1) of daily returns (6 decimal places)""",
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
        df = self.dedupe_stale_closes(df)

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
The data may contain quality issues such as duplicate rows, bad price ticks,
or unadjusted stock splits.

Cleaning rules:
1. Remove duplicate rows (keep first occurrence)
2. Remove bad price ticks: a tick is bad if its close is >50% different from
   BOTH the previous and next close (identifies spike outliers)
3. Detect 2-for-1 splits: look for a sustained ~50% drop (ratio between 0.4-0.6)
   that holds for at least 2 days (not a one-day spike). Exclude bad ticks.
   Then adjust all prices BEFORE the split by dividing by 2 (normalize to post-split basis).
4. Calculate total return as: (final_close - initial_close) / initial_close

Your task:
1. Identify and fix any data quality problems
2. Calculate the total return from first to last valid price
3. Report the final cleaned price

Return your answer as JSON with fields:
- problems: JSON array containing only the category codes that apply, from [duplicate, missing, bad_price, split] (for example ["duplicate"]); list each code at most once and include no others
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
        df = self.dedupe_stale_closes(df)

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
3. Calculate daily returns, then annualized volatility as: std(daily_returns, ddof=1) * sqrt(252)

Your task:
1. Identify and fix data quality problems
2. Calculate annualized volatility using the formula above (sample std with ddof=1)
3. Count the number of valid trading days after cleaning

Return your answer as JSON with fields:
- problems: JSON array containing only the category codes that apply, from [duplicate, missing, bad_price, split] (for example ["duplicate"]); list each code at most once and include no others
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
        2. Adjust for splits if present (divide pre-split by 2 to match post-split basis)
        3. Remove bad price ticks (outliers from both neighbors)

        Order matters: detect splits BEFORE filtering bad prices, so split day isn't removed.
        """
        df = df.copy()
        df = df.reset_index(drop=True)

        # 1. Remove duplicates
        df = df.drop_duplicates(subset=['date', 'ticker'], keep='first')
        df = df.reset_index(drop=True)

        # 2. Adjust for stock split FIRST (before filtering bad prices)
        if has_split:
            # Two-pass approach: identify bad ticks first, then detect splits ignoring bad ticks

            # Pass 1: Identify bad ticks (but don't remove yet)
            df['prev_close'] = df['close'].shift(1)
            df['next_close'] = df['close'].shift(-1)
            df['dev_from_prev'] = ((df['close'] - df['prev_close']) / df['prev_close']).abs()
            df['dev_from_next'] = ((df['close'] - df['next_close']) / df['next_close']).abs()
            df['is_bad_tick'] = (
                (df['dev_from_prev'] > 0.5) &
                (df['dev_from_next'] > 0.5) &
                df['prev_close'].notna() &
                df['next_close'].notna()
            )

            # Pass 2: Detect splits, computing ratios as if bad ticks were removed
            # Create a "clean" close series with bad ticks masked
            df['close_clean'] = df['close'].copy()
            df.loc[df['is_bad_tick'], 'close_clean'] = None
            df['close_clean_filled'] = df['close_clean'].ffill()  # Forward fill to skip bad ticks

            # Compute ratio using clean closes
            df['ratio_clean'] = df['close_clean_filled'] / df['close_clean_filled'].shift(1)

            # Also check if the move is sustained (next clean close also shows ~50% of pre-split)
            df['close_clean_2before'] = df['close_clean_filled'].shift(2)
            df['next_clean_close'] = df['close_clean'].shift(-1).bfill()  # Look ahead
            df['ratio_spanning'] = df['next_clean_close'] / df['close_clean_2before']

            # Split candidate: ratio between 0.4-0.6 AND sustained
            # Must be on a non-bad-tick row to be a proper split marker
            df['is_split_candidate'] = (
                (~df['is_bad_tick']) &
                (df['ratio_clean'] >= 0.4) &
                (df['ratio_clean'] <= 0.6) &
                (df['ratio_spanning'] >= 0.4) &
                (df['ratio_spanning'] <= 0.6)
            )

            split_indices = df[df['is_split_candidate']].index.tolist()

            if len(split_indices) > 0:
                split_pos = split_indices[0]

                # Adjust all prices BEFORE the split by dividing by 2
                if split_pos > 0:
                    df.loc[df.index < split_pos, ['open', 'high', 'low', 'close']] /= 2.0
                    df.loc[df.index < split_pos, ['open', 'high', 'low', 'close']] = df.loc[
                        df.index < split_pos, ['open', 'high', 'low', 'close']
                    ].round(2)
                    # Volume adjustment: multiply pre-split volume by 2
                    df.loc[df.index < split_pos, 'volume'] *= 2

            df = df.drop(columns=['prev_close', 'next_close', 'dev_from_prev', 'dev_from_next',
                                  'is_bad_tick', 'close_clean', 'close_clean_filled', 'ratio_clean',
                                  'close_clean_2before', 'next_clean_close', 'ratio_spanning',
                                  'is_split_candidate'])
            df = df.reset_index(drop=True)

        # 3. Remove bad price ticks AFTER split adjustment
        # A tick is bad if it's >50% away from BOTH previous AND next close
        # This correctly identifies spikes without removing the day after
        df['prev_close'] = df['close'].shift(1)
        df['next_close'] = df['close'].shift(-1)

        # Calculate deviations from neighbors
        df['dev_from_prev'] = ((df['close'] - df['prev_close']) / df['prev_close']).abs()
        df['dev_from_next'] = ((df['close'] - df['next_close']) / df['next_close']).abs()

        # A tick is bad if BOTH deviations are > 0.5
        # Keep first and last rows (they don't have both neighbors)
        df['is_bad_tick'] = (
            (df['dev_from_prev'] > 0.5) &
            (df['dev_from_next'] > 0.5) &
            df['prev_close'].notna() &
            df['next_close'].notna()
        )

        df = df[~df['is_bad_tick']]
        df = df.drop(columns=['prev_close', 'next_close', 'dev_from_prev', 'dev_from_next', 'is_bad_tick'])

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
        default=None,
        help='Output directory (default: tasks/ for easy, tasks_hard/ for hard)'
    )
    parser.add_argument(
        '--tier',
        type=str,
        choices=['easy', 'hard'],
        default='easy',
        help='Difficulty tier: easy (default) or hard'
    )

    args = parser.parse_args()

    # Warn if not using default seed 42 for eval set
    if args.seed != 42:
        print(f"\n⚠️  Using seed {args.seed} (not default 42)")
        print("   Default seed 42 is reserved for the evaluation set.")
        print("   Use different seeds for training data to avoid train/eval overlap.\n")

    if args.tier == 'hard':
        from src.hard_tier import HardTaskGenerator
        print("Generating HARD tier tasks...")
        generator = HardTaskGenerator(seed=args.seed)
    else:
        print("Generating EASY tier tasks...")
        generator = TaskGenerator(seed=args.seed)

    output_dir = args.output_dir or (Path('tasks_hard') if args.tier == 'hard' else Path('tasks'))
    tasks = generator.generate_all_tasks(output_dir, num_tasks=args.count)
    print(f"Created {len(tasks)} {args.tier} tier tasks with seed {args.seed}")


if __name__ == '__main__':
    main()

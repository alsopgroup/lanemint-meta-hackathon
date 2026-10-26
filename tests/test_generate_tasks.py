"""
Unit tests for the task generator.
"""


import pandas as pd
import pytest

from src.generate_tasks import TaskGenerator


class TestTaskGenerator:
    """Test suite for TaskGenerator."""

    def test_clean_prices_ohlc_consistency(self):
        """Test that generated prices have consistent OHLC relationships."""
        generator = TaskGenerator(seed=42)
        df = generator.generate_clean_prices('TEST', pd.Timestamp('2024-01-01'), 50)

        # Check all rows have valid OHLC relationships
        assert all(df['high'] >= df['low']), "High must be >= low"
        assert all(df['high'] >= df['open']), "High must be >= open"
        assert all(df['high'] >= df['close']), "High must be >= close"
        assert all(df['low'] <= df['open']), "Low must be <= open"
        assert all(df['low'] <= df['close']), "Low must be <= close"

    def test_cleaner_removes_only_intended_problems(self):
        """Test that _clean_data only removes rows with documented problems."""
        generator = TaskGenerator(seed=42)

        # Generate clean data
        df = generator.generate_clean_prices('TEST', pd.Timestamp('2024-01-01'), 50)
        original_len = len(df)

        # Clean it (should not remove anything)
        cleaned = generator._clean_data(df.copy(), has_split=False)

        # All rows should be kept since there are no problems
        assert len(cleaned) == original_len, \
            f"Clean data should not lose rows: {original_len} -> {len(cleaned)}"

    def test_cleaner_removes_duplicates(self):
        """Test that _clean_data removes duplicate rows."""
        generator = TaskGenerator(seed=42)
        df = generator.generate_clean_prices('TEST', pd.Timestamp('2024-01-01'), 50)

        # Add a duplicate
        df, _problem = generator.inject_duplicate_row(df)
        original_len = len(df)

        # Clean should remove the duplicate
        cleaned = generator._clean_data(df.copy(), has_split=False)
        assert len(cleaned) == original_len - 1, "Should remove duplicate row"

    def test_cleaner_removes_bad_prices(self):
        """Test that _clean_data removes bad price rows."""
        generator = TaskGenerator(seed=42)
        df = generator.generate_clean_prices('TEST', pd.Timestamp('2024-01-01'), 50)

        # Add a bad price
        df, _problem = generator.inject_bad_price(df)
        original_len = len(df)

        # Clean should remove the bad price row
        cleaned = generator._clean_data(df.copy(), has_split=False)
        assert len(cleaned) < original_len, "Should remove bad price row"

    def test_cleaner_adjusts_splits_correctly(self):
        """Test that _clean_data correctly adjusts for stock splits."""
        generator = TaskGenerator(seed=42)
        df = generator.generate_clean_prices('TEST', pd.Timestamp('2024-01-01'), 60)

        # Inject a split
        df, _problem = generator.inject_stock_split(df)

        # Record a price before the split
        pre_split_idx = 10
        df.iloc[pre_split_idx]['close']

        # Clean with split adjustment
        cleaned = generator._clean_data(df.copy(), has_split=True)

        # After cleaning, pre-split prices should be doubled (roughly)
        # The adjustment multiplies by 2, so check it happened
        post_split_idx = len(cleaned) - 10
        pre_split_idx_cleaned = min(10, len(cleaned) - 20)

        if pre_split_idx_cleaned >= 0 and post_split_idx < len(cleaned):
            # Pre-split and post-split prices should be similar after adjustment
            pre_price = cleaned.iloc[pre_split_idx_cleaned]['close']
            post_price = cleaned.iloc[post_split_idx]['close']

            # They should be within same order of magnitude (both adjusted to post-split scale)
            ratio = pre_price / post_price
            assert 0.5 < ratio < 2.0, \
                f"Split adjustment failed: pre={pre_price}, post={post_price}, ratio={ratio}"

    def test_cleaner_keeps_first_row(self):
        """Test that _clean_data does not drop the first row."""
        generator = TaskGenerator(seed=42)
        df = generator.generate_clean_prices('TEST', pd.Timestamp('2024-01-01'), 50)

        first_date = df.iloc[0]['date']

        # Clean it
        cleaned = generator._clean_data(df.copy(), has_split=False)

        # First row should be kept
        assert cleaned.iloc[0]['date'] == first_date, "First row should not be dropped"

    def test_seed_reproducibility(self):
        """Test that same seed produces same tasks."""
        gen1 = TaskGenerator(seed=123)
        task1, df1 = gen1.generate_task_return_calculation()

        gen2 = TaskGenerator(seed=123)
        task2, df2 = gen2.generate_task_return_calculation()

        # Same seed should produce identical data
        assert len(df1) == len(df2)
        assert df1.iloc[0]['close'] == df2.iloc[0]['close']

        # Expected values should match (they might differ in task_id but not in expected results)
        # The key is that the actual data and calculations are the same
        assert task1['expected']['mean_return'] == task2['expected']['mean_return']
        assert task1['expected']['std_return'] == task2['expected']['std_return']

    def test_different_seeds_different_tasks(self):
        """Test that different seeds produce different tasks."""
        gen1 = TaskGenerator(seed=123)
        gen2 = TaskGenerator(seed=456)

        task1, _df1 = gen1.generate_task_return_calculation()
        task2, _df2 = gen2.generate_task_return_calculation()

        # Different seeds should produce different results
        assert task1['expected'] != task2['expected']

    def test_no_train_eval_overlap(self):
        """Test that training seeds don't overlap with eval seed 42."""
        eval_gen = TaskGenerator(seed=42)
        train_gen = TaskGenerator(seed=100)

        # Generate eval set
        eval_tasks = []
        for _ in range(5):
            task, _df = eval_gen.generate_task_return_calculation()
            eval_tasks.append(task['expected'])

        # Generate training set with different seed
        train_tasks = []
        for _ in range(5):
            task, _df = train_gen.generate_task_return_calculation()
            train_tasks.append(task['expected'])

        # No overlap in expected values
        eval_set = {str(t) for t in eval_tasks}
        train_set = {str(t) for t in train_tasks}

        overlap = eval_set.intersection(train_set)
        assert len(overlap) == 0, f"Found {len(overlap)} overlapping tasks between eval and train"


class TestCleaningRuleConsistency:
    """Test that prompts match cleaning implementation."""

    def test_return_calculation_prompt_matches_cleaner(self):
        """Test that return calculation prompt describes the actual cleaning logic."""
        generator = TaskGenerator(seed=42)
        task, _df = generator.generate_task_return_calculation()

        prompt = task['prompt']

        # Check that prompt mentions key cleaning rules
        assert 'duplicate' in prompt.lower(), "Prompt should mention duplicate handling"
        assert '50%' in prompt, "Prompt should mention 50% threshold"
        assert 'pct_change' in prompt or 'daily returns' in prompt.lower(), \
            "Prompt should explain return calculation"

    def test_backtest_prompt_mentions_split_adjustment(self):
        """Test that backtest prompt mentions split adjustment."""
        generator = TaskGenerator(seed=42)

        # Keep generating until we get one with a split
        for _ in range(20):
            task, _df = generator.generate_task_backtest()
            prompt = task['prompt']
            if 'split' in prompt.lower():
                # Check it explains the adjustment
                assert 'adjust' in prompt.lower() or 'multiply' in prompt.lower(), \
                    "Backtest prompt should explain split adjustment method"
                break


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

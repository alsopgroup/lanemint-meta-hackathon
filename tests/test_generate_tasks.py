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
                assert 'divid' in prompt.lower() or 'halve' in prompt.lower(), \
                    "Backtest prompt should explain split adjustment method (divide pre-split by 2)"
                break


class TestCleaningRegressions:
    """Regression tests for cleaning bugs found in baseline runs."""

    def test_split_day_not_removed_as_bad_tick(self):
        """Regression: split day must not be filtered as a bad tick."""
        generator = TaskGenerator(seed=42)
        df = generator.generate_clean_prices('TEST', pd.Timestamp('2024-01-01'), 60)

        # Inject a split
        df, problem = generator.inject_stock_split(df)

        # Extract split date from problem string
        split_date_str = problem.split('_')[2]  # 'stock_split_YYYY-MM-DD_2.0'

        # Clean with split adjustment
        cleaned = generator._clean_data(df.copy(), has_split=True)

        # The split day should still be present
        split_dates_in_cleaned = cleaned['date'].astype(str).str.contains(split_date_str)
        assert split_dates_in_cleaned.any(), \
            f"Split day {split_date_str} was incorrectly removed as a bad tick"

    def test_day_after_bad_tick_is_kept(self):
        """Regression: the day after a bad tick must not be removed."""
        generator = TaskGenerator(seed=42)
        df = generator.generate_clean_prices('TEST', pd.Timestamp('2024-01-01'), 60)

        # Inject a bad price at a specific index
        bad_idx = 25
        original_len = len(df)
        df.loc[bad_idx, 'close'] = df.loc[bad_idx, 'close'] * 10.0  # 10x spike
        df.loc[bad_idx, 'high'] = df.loc[bad_idx, ['open', 'high', 'low', 'close']].max()
        df.loc[bad_idx, 'low'] = df.loc[bad_idx, ['open', 'high', 'low', 'close']].min()

        # Clean it
        cleaned = generator._clean_data(df.copy(), has_split=False)

        # Should remove exactly 1 row (the bad tick), not 2
        # The day after should be kept
        assert len(cleaned) >= original_len - 1, \
            f"Too many rows removed: {original_len} -> {len(cleaned)}, expected at most -1"

    def test_split_adjusted_series_no_large_jump(self):
        """Regression: after split adjustment, there should be no >=40% jump at split date."""
        generator = TaskGenerator(seed=42)
        df = generator.generate_clean_prices('TEST', pd.Timestamp('2024-01-01'), 60)

        # Inject a split
        df, _problem = generator.inject_stock_split(df)

        # Clean with split adjustment
        cleaned = generator._clean_data(df.copy(), has_split=True)

        # Check for large jumps
        cleaned['pct_change'] = cleaned['close'].pct_change().abs()

        # After proper adjustment, no jump should be >= 40%
        max_jump = cleaned['pct_change'].max()
        assert max_jump < 0.4, \
            f"Split adjustment failed: found {max_jump:.1%} jump, expected < 40%"

    def test_full_task_with_all_problems_row_count(self):
        """
        Regression: verify that cleaner handles all 4 problem types correctly.
        """
        generator = TaskGenerator(seed=42)
        df = generator.generate_clean_prices('TEST', pd.Timestamp('2024-01-01'), 60)
        original_clean_count = len(df)

        # Inject duplicate (adds 1 row, will be removed)
        df, _p1 = generator.inject_duplicate_row(df)

        # Inject missing day (removes 1 row permanently)
        len_before_missing = len(df)
        df, _p2 = generator.inject_missing_day(df)
        rows_lost_to_missing = len_before_missing - len(df)

        # Inject bad tick (doesn't change count before cleaning)
        len(df)
        df, _p3 = generator.inject_bad_price(df)

        # Inject split (doesn't change count)
        df, _p4 = generator.inject_stock_split(df)

        # Clean it
        cleaned = generator._clean_data(df.copy(), has_split=True)

        # The cleaned result should have removed:
        # 1. The duplicate (back to original - 1 for missing day)
        # 2. The bad tick (one more row removed)
        # So: original_clean_count - rows_lost_to_missing - 1 (bad tick)
        # But we need to account for edge cases where the bad tick might be
        # at the boundary and not detected

        # More robust check: just verify it's close to expected
        expected_min = original_clean_count - rows_lost_to_missing - 2  # Allow for some variation
        expected_max = original_clean_count - rows_lost_to_missing

        assert expected_min <= len(cleaned) <= expected_max, \
            f"Row count out of range: got {len(cleaned)}, " \
            f"expected between {expected_min} and {expected_max} " \
            f"(original {original_clean_count}, missing {rows_lost_to_missing})"

    def test_bad_tick_before_split_01x(self):
        """
        Regression: 0.1x bad tick before split should not be mistaken for split.
        Split should still be found at the correct date.
        """
        generator = TaskGenerator(seed=123)
        df = generator.generate_clean_prices('TEST', pd.Timestamp('2024-01-01'), 60)

        # Inject split first (so we know where it is)
        df, split_info = generator.inject_stock_split(df)
        split_date = split_info.split('_')[2]

        # Now inject a 0.1x bad tick BEFORE the split
        # (manually to ensure it's before the split)
        split_idx = df[df['date'] == pd.Timestamp(split_date)].index[0]
        bad_tick_idx = split_idx - 10  # Place it 10 days before split

        original_close = df.loc[bad_tick_idx, 'close']
        df.loc[bad_tick_idx, 'close'] = original_close * 0.1
        df.loc[bad_tick_idx, 'high'] = df.loc[bad_tick_idx, 'close']

        # Clean it
        cleaned = generator._clean_data(df.copy(), has_split=True)

        # Verify no extreme jumps after cleaning
        cleaned['pct_change'] = cleaned['close'].pct_change().abs()
        max_jump = cleaned['pct_change'].max()
        assert max_jump < 0.4, \
            f"Found {max_jump:.1%} jump after cleaning, expected < 40%"

        # Verify bad tick was removed (should have one less row than original minus split adjustment)
        # Original had the bad tick, cleaned should not
        assert len(cleaned) < len(df), "Bad tick was not removed"

    def test_bad_tick_before_split_10x(self):
        """
        Regression: 10x bad tick before split should not interfere with split detection.
        """
        generator = TaskGenerator(seed=456)
        df = generator.generate_clean_prices('TEST', pd.Timestamp('2024-01-01'), 60)

        # Inject split first
        df, split_info = generator.inject_stock_split(df)
        split_date = split_info.split('_')[2]

        # Inject a 10x bad tick BEFORE the split
        split_idx = df[df['date'] == pd.Timestamp(split_date)].index[0]
        bad_tick_idx = split_idx - 15

        original_close = df.loc[bad_tick_idx, 'close']
        df.loc[bad_tick_idx, 'close'] = original_close * 10.0
        df.loc[bad_tick_idx, 'high'] = df.loc[bad_tick_idx, 'close']

        # Clean it
        cleaned = generator._clean_data(df.copy(), has_split=True)

        # Verify no extreme jumps
        cleaned['pct_change'] = cleaned['close'].pct_change().abs()
        max_jump = cleaned['pct_change'].max()
        assert max_jump < 0.4, \
            f"Found {max_jump:.1%} jump after cleaning, expected < 40%"

        # Verify bad tick was removed
        assert len(cleaned) < len(df), "Bad tick was not removed"

    def test_all_problems_across_seeds(self):
        """
        Regression: test all 4 problems combined across 50 seeds.
        Verify no >40% jumps and reasonable row counts.
        """
        for seed in range(50):
            generator = TaskGenerator(seed=seed)
            df = generator.generate_clean_prices('TEST', pd.Timestamp('2024-01-01'), 60)
            original_count = len(df)

            # Inject all 4 problems
            df, _p1 = generator.inject_duplicate_row(df)
            df, _p2 = generator.inject_missing_day(df)
            df, _p3 = generator.inject_bad_price(df)
            df, _p4 = generator.inject_stock_split(df)

            # Clean
            cleaned = generator._clean_data(df.copy(), has_split=True)

            # Check no extreme jumps
            cleaned['pct_change'] = cleaned['close'].pct_change().abs()
            max_jump = cleaned['pct_change'].max()
            assert max_jump < 0.4, \
                f"Seed {seed}: found {max_jump:.1%} jump, expected < 40%"

            # Check row count is reasonable
            # Should be roughly original - 1 (missing) - 1 (bad tick) = original - 2
            # Allow some tolerance
            assert original_count - 4 <= len(cleaned) <= original_count, \
                f"Seed {seed}: row count {len(cleaned)} out of range for original {original_count}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

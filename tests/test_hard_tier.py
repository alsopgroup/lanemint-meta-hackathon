"""
Unit tests for the hard-tier task generator.
"""

from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from src.hard_tier import HardTaskGenerator


class TestHardTaskGenerator:
    """Test suite for HardTaskGenerator."""

    def test_hard_tier_reproducibility(self):
        """Test that hard tasks are reproducible with same seed."""
        gen1 = HardTaskGenerator(seed=42)
        task1, df1 = gen1.generate_task_max_drawdown()

        gen2 = HardTaskGenerator(seed=42)
        task2, df2 = gen2.generate_task_max_drawdown()

        # Task structure should be the same
        assert task1['id'] == task2['id']
        assert task1['type'] == task2['type']
        # Data should be identical
        pd.testing.assert_frame_equal(df1, df2)
        # Expected answers should match
        assert task1['expected'] == task2['expected']

    def test_hard_tasks_use_h_prefix(self):
        """Test that hard tasks use 'task_h_' prefix to avoid ID conflicts."""
        generator = HardTaskGenerator(seed=42)
        task, _df = generator.generate_task_max_drawdown()

        assert task['id'].startswith('task_h_')
        assert 'task_h_' in task['data_file']

    def test_no_off_limits_problem_types(self):
        """Test that hard tasks don't use forbidden problem types."""
        generator = HardTaskGenerator(seed=42)

        off_limits = [
            'delisted', 'survivorship', 'ticker_rename',
            'dividend', 'stale_price', 'repeated_price',
            'too_short', 'insufficient_history'
        ]

        # Generate a few tasks and check problem descriptions
        for _i in range(10):
            task, _df = generator.generate_task_max_drawdown()
            problems = task['expected']['problems_found']

            for problem in problems:
                problem_lower = problem.lower()
                for forbidden in off_limits:
                    assert forbidden not in problem_lower, \
                        f"Found forbidden problem type '{forbidden}' in: {problem}"

    def test_hard_tier_answer_reproducibility(self):
        """Test that reference answers are reproducible from same starting state."""
        # Generate same task with same seed - should get identical answers
        gen1 = HardTaskGenerator(seed=100)
        task1, _df1 = gen1.generate_task_max_drawdown()

        gen2 = HardTaskGenerator(seed=100)
        task2, _df2 = gen2.generate_task_max_drawdown()

        # Expected answers should match exactly
        assert task1['expected'] == task2['expected']

    def test_hard_tier_across_seeds(self):
        """Test hard tier generation across 50 seeds for consistency."""
        for seed in range(50):
            generator = HardTaskGenerator(seed=seed)
            task, df = generator.generate_task_max_drawdown()

            # Basic structure checks
            assert 'id' in task
            assert 'type' in task
            assert 'prompt' in task
            assert 'expected' in task
            assert 'problems_found' in task['expected']
            assert 'max_drawdown' in task['expected']

            # Data checks
            assert len(df) > 0
            assert 'date' in df.columns or 'timestamp' in df.columns
            assert 'close' in df.columns

            # Answer should be a valid number
            assert isinstance(task['expected']['max_drawdown'], (int, float))
            assert task['expected']['max_drawdown'] <= 0  # Drawdown is negative

    def test_multi_asset_task_has_multiple_tickers(self):
        """Test that multi-asset tasks actually have multiple tickers."""
        generator = HardTaskGenerator(seed=42)
        task, df = generator.generate_task_multi_asset_alignment()

        assert 'ticker' in df.columns
        num_tickers = df['ticker'].nunique()
        assert num_tickers >= 3, f"Expected >= 3 tickers, got {num_tickers}"

        # Check expected fields
        assert 'portfolio_return' in task['expected']
        assert 'num_assets' in task['expected']
        assert task['expected']['num_assets'] >= 3

    def test_prompts_dont_list_problems(self):
        """Test that hard tier prompts don't explicitly list which problems are present."""
        generator = HardTaskGenerator(seed=42)

        task1, _df1 = generator.generate_task_max_drawdown()
        task2, _df2 = generator.generate_task_multi_asset_alignment()

        # Prompts should say "may contain" not "contains" or list specific problems
        for task in [task1, task2]:
            prompt_lower = task['prompt'].lower()
            assert 'may contain' in prompt_lower or 'data may' in prompt_lower

            # Should not enumerate specific problems
            problems = task['expected']['problems_found']
            for problem in problems:
                # Extract the problem type (e.g., "out_of_order" from "out_of_order_rows_3_swaps")
                problem.split('_')[0:3]  # Get first few words
                # Prompt shouldn't explicitly mention this specific problem instance
                assert problem not in task['prompt'], \
                    f"Prompt should not explicitly list problem: {problem}"

    def test_prompts_specify_computation_rules(self):
        """Test that prompts explicitly state computation rules."""
        generator = HardTaskGenerator(seed=42)

        task1, _df1 = generator.generate_task_max_drawdown()
        assert 'drawdown' in task1['prompt'].lower()
        assert 'cummax' in task1['prompt'].lower() or 'maximum' in task1['prompt'].lower()

        task2, _df2 = generator.generate_task_multi_asset_alignment()
        assert 'equal-weight' in task2['prompt'].lower() or 'equal weight' in task2['prompt'].lower()
        assert 'rebalanc' in task2['prompt'].lower()

    def test_hard_tier_default_count(self):
        """Test that hard tier can generate default 25 tasks."""
        generator = HardTaskGenerator(seed=100)
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            tasks = generator.generate_all_tasks(Path(tmpdir), num_tasks=25)
            assert len(tasks) == 25

            # Check all tasks have unique IDs
            task_ids = [t['id'] for t in tasks]
            assert len(set(task_ids)) == len(task_ids)


class TestHardTierProblems:
    """Test specific hard-tier problem types."""

    def test_inject_unit_change(self):
        """Test unit change injection (dollars to cents)."""
        generator = HardTaskGenerator(seed=42)
        dfs = generator.generate_clean_prices_multi_asset(['TEST'], datetime(2024, 1, 1), 60)
        df = dfs['TEST']

        original_close = df.iloc[30]['close']
        df_modified, problem = generator.inject_unit_change(df, 30)

        # After change point, prices should be 100x higher
        assert df_modified.iloc[35]['close'] > original_close * 50
        assert 'unit_change' in problem

    def test_inject_reverse_split(self):
        """Test reverse split injection."""
        generator = HardTaskGenerator(seed=42)
        dfs = generator.generate_clean_prices_multi_asset(['TEST'], datetime(2024, 1, 1), 60)
        df = dfs['TEST']

        original_close = df.iloc[30]['close']
        df_modified, problem = generator.inject_reverse_split(df, ratio=0.1)

        # After split, prices should be ~10x higher
        assert df_modified.iloc[35]['close'] > original_close * 5
        assert 'reverse_split' in problem or 'split' in problem

    def test_inject_out_of_order(self):
        """Test out-of-order row injection."""
        generator = HardTaskGenerator(seed=42)
        dfs = generator.generate_clean_prices_multi_asset(['TEST'], datetime(2024, 1, 1), 60)
        df = dfs['TEST']

        original_order = df['date'].tolist()
        df_modified, problem = generator.inject_out_of_order_rows(df, num_swaps=3)
        new_order = df_modified['date'].tolist()

        # Some dates should be out of order
        assert original_order != new_order
        assert 'out_of_order' in problem

    def test_max_drawdown_computation(self):
        """Test max drawdown calculation."""
        generator = HardTaskGenerator(seed=42)

        # Simple test case: prices go 100 -> 150 -> 75 -> 120
        prices = pd.Series([100, 150, 75, 120])
        max_dd = generator.compute_max_drawdown(prices)

        # Peak is 150, trough after peak is 75, drawdown = (75-150)/150 = -0.5
        assert max_dd < -0.45  # Account for floating point
        assert max_dd > -0.55


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

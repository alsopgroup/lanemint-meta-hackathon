"""
Unit tests for the task scorer.
"""

from pathlib import Path

import pytest

from src.scorer import TaskScorer


class TestTaskScorer:
    """Test suite for TaskScorer."""

    def test_parse_response_plain_json(self):
        """Test parsing plain JSON response."""
        scorer = TaskScorer(Path('tasks/answers'))
        response = '{"mean_return": 0.001, "std_return": 0.02}'
        parsed = scorer.parse_response(response)
        assert parsed == {"mean_return": 0.001, "std_return": 0.02}

    def test_parse_response_markdown_json(self):
        """Test parsing JSON in markdown code block."""
        scorer = TaskScorer(Path('tasks/answers'))
        response = '```json\n{"mean_return": 0.001}\n```'
        parsed = scorer.parse_response(response)
        assert parsed == {"mean_return": 0.001}

    def test_parse_response_markdown_plain(self):
        """Test parsing JSON in plain markdown code block."""
        scorer = TaskScorer(Path('tasks/answers'))
        response = '```\n{"mean_return": 0.001}\n```'
        parsed = scorer.parse_response(response)
        assert parsed == {"mean_return": 0.001}

    def test_parse_response_invalid_json(self):
        """Test handling of invalid JSON."""
        scorer = TaskScorer(Path('tasks/answers'))
        response = 'This is not JSON'
        parsed = scorer.parse_response(response)
        assert 'error' in parsed

    def test_compare_float_within_tolerance(self):
        """Test float comparison within tolerance."""
        scorer = TaskScorer(Path('tasks/answers'))
        assert scorer.compare_float(0.001234, 0.001235, tolerance=1e-5)

    def test_compare_float_outside_tolerance(self):
        """Test float comparison outside tolerance."""
        scorer = TaskScorer(Path('tasks/answers'))
        assert not scorer.compare_float(0.001234, 0.002000, tolerance=1e-5)

    def test_score_problems_found_exact_match(self):
        """Test problem scoring with exact match."""
        scorer = TaskScorer(Path('tasks/answers'))
        actual = ['duplicate row on 2024-01-15', 'bad price on 2024-02-20']
        expected = ['duplicate_row_2024-01-15', 'bad_price_2024-02-20_10.0']
        score, feedback = scorer.score_problems_found(actual, expected)
        assert score == 1.0

    def test_score_problems_found_partial_match(self):
        """Test problem scoring with partial match."""
        scorer = TaskScorer(Path('tasks/answers'))
        actual = ['duplicate row detected']
        expected = ['duplicate_row_2024-01-15', 'bad_price_2024-02-20']
        score, feedback = scorer.score_problems_found(actual, expected)
        assert 0 < score < 1.0
        assert 'Missed' in feedback

    def test_score_problems_found_no_problems(self):
        """Test problem scoring when none found."""
        scorer = TaskScorer(Path('tasks/answers'))
        actual = []
        expected = ['duplicate_row_2024-01-15']
        score, feedback = scorer.score_problems_found(actual, expected)
        assert score == 0.0
        assert feedback == "No problems identified"

    def test_score_problems_found_extra_problems(self):
        """Test problem scoring with false positives."""
        scorer = TaskScorer(Path('tasks/answers'))
        actual = ['duplicate', 'split', 'outlier']
        expected = ['duplicate_row_2024-01-15']
        score, feedback = scorer.score_problems_found(actual, expected)
        assert 'False positives' in feedback


class TestProblemDetection:
    """Test problem detection logic."""

    def test_detect_duplicate_keyword(self):
        """Test detection of duplicate keyword."""
        problems = ['found duplicate row', 'duplicate entry at line 10']
        assert any('duplicate' in p.lower() for p in problems)

    def test_detect_missing_keyword(self):
        """Test detection of missing/gap keyword."""
        problems = ['missing day between dates', 'gap in data']
        assert any('missing' in p.lower() or 'gap' in p.lower() for p in problems)

    def test_detect_bad_price_keyword(self):
        """Test detection of bad price keyword."""
        problems = ['bad price on 2024-01-15', 'incorrect value', 'price spike']
        assert any(
            'bad' in p.lower() or 'incorrect' in p.lower() or 'spike' in p.lower()
            for p in problems
        )

    def test_detect_split_keyword(self):
        """Test detection of stock split keyword."""
        problems = ['stock split detected', '2-for-1 split']
        assert any('split' in p.lower() for p in problems)

    def test_negative_statements_not_counted(self):
        """Test that negative statements don't count as problem detections."""
        scorer = TaskScorer(Path('tasks/answers'))

        # Case 1: "No duplicate" should not match duplicate
        actual = ['No duplicate rows found', 'Data looks clean']
        expected = ['duplicate_row_2024-01-15']
        score, _feedback = scorer.score_problems_found(actual, expected)
        assert score == 0.0, "Negative statement should not count as detection"

        # Case 2: "No missing" should not match missing
        actual = ['No missing days', 'Complete dataset']
        expected = ['missing_day_2024-01-20']
        score, _feedback = scorer.score_problems_found(actual, expected)
        assert score == 0.0

        # Case 3: "Not a split" should not match split
        actual = ['This is not a split issue']
        expected = ['stock_split_2024-02-01_2.0']
        score, _feedback = scorer.score_problems_found(actual, expected)
        assert score == 0.0

    def test_positive_detections_still_work(self):
        """Test that actual problem mentions still work after negation fix."""
        scorer = TaskScorer(Path('tasks/answers'))

        actual = ['Found duplicate row on 2024-01-15']
        expected = ['duplicate_row_2024-01-15']
        score, _feedback = scorer.score_problems_found(actual, expected)
        assert score == 1.0


# Phrases supplied by the PR #6 reviewer (neg.py).
NEGATION_CASES = [
    ('no duplicates', set()), ('there were no duplicate rows', set()),
    ('found 2 duplicate rows, no missing days', {'duplicate'}),
    ('not a bad tick', set()), ('duplicates: none', set()),
    ('No duplicate rows found', set()), ('No missing days', set()),
    ('This is not a split issue', set()), ('no evidence of any duplicate rows', set()),
    ('checked for duplicates but found none', set()), ('nothing missing', set()),
    ("wasn't a split", set()),
    ('no split detected; 1-for-3 reverse split found on 2024-02-16', {'split'}),
    ('duplicate row on 2024-01-15 (not a missing day)', {'duplicate'}),
    ('missing day 2024-01-20, no duplicate', {'missing'}),
    ('known issue: 2024-01-15 duplicate', {'duplicate'}),
    ('cannot rule out a 3:1 reverse split', set()),
    ('prices badly scaled after 2024-03-04', {'bad_price'}),
    ('jumped 100x: unit change to cents', {'bad_price'}),
    ('timestamp gap of 3 days', {'missing'}), ('nonduplicated', set()),
    ('zero-volume bar, no gap', set()), ('NO DUPLICATES', set()),
]


class TestNegationPhrases:
    @pytest.mark.parametrize(('phrase', 'want'), NEGATION_CASES)
    def test_phrase(self, phrase, want):
        from src.scorer import categories_for_text
        got = categories_for_text(phrase) & {'duplicate', 'missing', 'bad_price', 'split'}
        assert got == want

    def test_hard_categories(self):
        from src.scorer import categories_for_text
        assert 'unit_change' in categories_for_text('prices switch from dollars to cents')
        assert 'out_of_order' in categories_for_text('rows out of chronological order')
        assert 'reverse_split' in categories_for_text('1-for-3 reverse split')
        assert 'out_of_order' not in categories_for_text('no out of order issues')


class TestHardNumericScoring:
    def _scorer_with(self, tmp_path, answer):
        import json as _json
        (tmp_path / 'task_h_1001_answer.json').write_text(_json.dumps(answer))
        return TaskScorer(tmp_path)

    def test_max_drawdown_scored(self, tmp_path):
        import json as _json
        scorer = self._scorer_with(tmp_path, {'max_drawdown': -0.20, 'num_clean_days': 10,
                                               'problems_found': ['unit_change_x']})
        good = scorer.score_task('task_h_1001', _json.dumps(
            {'max_drawdown': -0.20, 'num_clean_days': 10,
             'problems_found': ['unit change to cents']}))
        wrong = scorer.score_task('task_h_1001', _json.dumps(
            {'max_drawdown': -1.40, 'num_clean_days': 10, 'problems_found': ['junk']}))
        assert good['score'] == 1.0
        assert wrong['score'] < 1.0
        assert wrong['component_scores']['max_drawdown'] == 0.0
        assert wrong['component_scores']['problems'] == 0.0

    def test_portfolio_fields_scored(self, tmp_path):
        import json as _json
        scorer = self._scorer_with(tmp_path, {'portfolio_return': 0.05, 'num_assets': 3,
                                              'num_aligned_days': 40,
                                              'problems_found': ['reverse_split_x']})
        perfect = scorer.score_task('task_h_1001', _json.dumps(
            {'portfolio_return': 0.05, 'num_assets': 3, 'num_aligned_days': 40,
             'problems_found': ['1-for-3 reverse split']}))
        assert perfect['score'] == 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

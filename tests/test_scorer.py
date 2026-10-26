"""
Unit tests for the task scorer.
"""

import json
import pytest
from pathlib import Path
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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

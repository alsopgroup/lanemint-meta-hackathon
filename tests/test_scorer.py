"""
Unit tests for the task scorer.
"""

from pathlib import Path

import pytest

from src.scorer import TaskScorer


class TestKeyStrings:
    """Answer-key strings must map to categories without negation parsing."""

    def test_bad_price_key_string_keeps_its_code(self):
        from src.scorer import categories_for_items, categories_for_key_items
        key = ['bad_price_2024-07-09_0.1']
        assert categories_for_key_items(key) == {'bad_price'}
        # the perfect-answer path: the real key strings as the response
        assert categories_for_items(key) == {'bad_price'}
        scorer = TaskScorer(Path('tasks/answers'))
        score, _ = scorer.score_problems_found(['bad_price'], key)
        assert score == 1.0
        score, _ = scorer.score_problems_found(key, key)
        assert score == 1.0

    def test_zero_counts_negate_but_decimals_do_not(self):
        from src.scorer import categories_for_items, categories_for_text
        # explicit none/0 counts in a model answer mean "not detected"
        assert categories_for_text('There are 0 duplicate rows') == set()
        assert categories_for_text('0 duplicates found') == set()
        assert categories_for_items(['missing: 0']) == set()
        assert categories_for_items(['missing: 0 days']) == set()
        assert categories_for_items(['duplicate_rows: none']) == set()
        # decimals are not zero counts, and key strings keep their codes
        assert categories_for_text('0.1x bad price spike') == {'bad_price'}
        from src.scorer import categories_for_key_items
        assert categories_for_key_items(['bad_price_2024-07-09_0.1']) == {'bad_price'}
        assert categories_for_key_items(['bad_price_2024-07-09_10.0']) == {'bad_price'}
        assert categories_for_key_items(['stock_split_2024-07-09_2.0']) == {'split'}

    def test_every_key_scores_perfectly_seeds_1_to_200_both_tiers(self):
        import json
        import tempfile

        from src.generate_tasks import TaskGenerator
        from src.hard_tier import HardTaskGenerator
        scorer = TaskScorer(Path('tasks/answers'))
        for seed in range(1, 201):
            for gen in (HardTaskGenerator(seed=seed), TaskGenerator(seed=seed)):
                with tempfile.TemporaryDirectory() as tmp:
                    tasks = gen.generate_all_tasks(Path(tmp), num_tasks=5)
                    for t in tasks:
                        key = json.loads((Path(tmp) / 'answers'
                                          / f"{t['id']}_answer.json").read_text())
                        strings = key['problems_found']
                        score, fb = scorer.score_problems_found(strings, strings)
                        assert score == 1.0, (seed, t['id'], strings, fb)

    def test_seed42_named_easy_tasks_score_perfectly(self):
        import json
        import tempfile

        from src.generate_tasks import TaskGenerator
        scorer = TaskScorer(Path('tasks/answers'))
        with tempfile.TemporaryDirectory() as tmp:
            tasks = TaskGenerator(seed=42).generate_all_tasks(Path(tmp), num_tasks=25)
            by_id = {t['id']: t for t in tasks}
            for task_id in ('task_013', 'task_018', 'task_023'):
                key = json.loads((Path(tmp) / 'answers'
                                  / f"{task_id}_answer.json").read_text())
                assert 'bad_price' in str(key['problems_found']), task_id
                score, fb = scorer.score_problems_found(
                    key['problems_found'], key['problems_found'])
                assert score == 1.0, (task_id, fb)
                assert by_id[task_id]['id'] == task_id


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
    ('no split detected; 1-for-3 reverse split found on 2024-02-16', {'reverse_split'}),
    ('duplicate row on 2024-01-15 (not a missing day)', {'duplicate'}),
    ('missing day 2024-01-20, no duplicate', {'missing'}),
    ('known issue: 2024-01-15 duplicate', {'duplicate'}),
    ('cannot rule out a 3:1 reverse split', set()),
    ('prices badly scaled after 2024-03-04', {'bad_price'}),
    ('jumped 100x: unit change to cents', {'unit_change'}),
    ('timestamp gap of 3 days', {'missing'}), ('nonduplicated', set()),
    ('zero-volume bar, no gap', set()), ('NO DUPLICATES', set()),
]


class TestNegationPhrases:
    @pytest.mark.parametrize(('phrase', 'want'), NEGATION_CASES)
    def test_phrase(self, phrase, want):
        from src.scorer import categories_for_text
        got = categories_for_text(phrase)
        assert got == want

    def test_hard_categories(self):
        from src.scorer import categories_for_text
        assert 'unit_change' in categories_for_text('prices switch from dollars to cents')
        assert 'out_of_order' in categories_for_text('rows out of chronological order')
        assert 'reverse_split' in categories_for_text('1-for-3 reverse split')
        assert 'out_of_order' not in categories_for_text('no out of order issues')


NEW_PHRASES_POSITIVE = [
    ('out_of_order: rows are not sorted by date', 'out_of_order'),
    ('adj_close: adj_close does not match close after the split', 'adj_close'),
    ('currency: ACME prices are in EUR not USD', 'currency'),
    ('column_order: columns are not in the original order', 'column_order'),
]
NEW_PHRASES_NEGATIVE = [
    'free of duplicates', 'duplicate-free',
    'lacks duplicates', 'absence of missing days', 'unit change ruled out',
    'possible split (unconfirmed)', 'neither duplicates nor gaps',
]


class TestNewNegationPhrases:
    @pytest.mark.parametrize(('phrase', 'category'), NEW_PHRASES_POSITIVE)
    def test_structured_positive(self, phrase, category):
        from src.scorer import categories_for_items
        assert category in categories_for_items([phrase])

    @pytest.mark.parametrize('phrase', NEW_PHRASES_NEGATIVE)
    def test_negative(self, phrase):
        from src.scorer import categories_for_text
        assert categories_for_text(phrase) == set()

    def test_besides_and_instead(self):
        from src.scorer import categories_for_text
        # A repeated instant is the timezone category, not a duplicate row.
        assert categories_for_text('No issues besides a duplicate instant') == {'timezone'}
        got = categories_for_text('no reverse split - prices are in cents instead')
        assert 'unit_change' in got and 'split' not in got and 'reverse_split' not in got

    def test_code_none_and_but_missing(self):
        from src.scorer import categories_for_items, categories_for_text
        assert 'split' not in categories_for_items(['split: none detected'])
        assert 'reverse_split' not in categories_for_items(['reverse_split: not present'])
        got = categories_for_text('not a duplicate but a missing day')
        assert 'missing' in got and 'duplicate' not in got

    def test_code_na_nothing_and_emdash_negatives(self):
        from src.scorer import categories_for_items, categories_for_text
        assert categories_for_items(['duplicate: n/a']) == set()
        assert categories_for_items(['missing: nothing missing']) == set()
        assert categories_for_items(['missing — none']) == set()
        assert categories_for_text('timestamps mix utc and offsets') == {'timezone'}


class TestSplitRatioMapping:
    """Only 1-for-N (or explicit 'reverse') is a reverse split; N-for-1 is a split."""

    @pytest.mark.parametrize(('phrase', 'want'), [
        ('2-for-1 split', {'split'}),
        ('2:1 split', {'split'}),
        ('10-for-1 forward split', {'split'}),
        ('stock split (2:1)', {'split'}),
        ('1-for-3 reverse split', {'reverse_split'}),
        ('1-for-10 split', {'reverse_split'}),
        ('1:5 reverse split', {'reverse_split'}),
        ('3:1 reverse split', {'reverse_split'}),
    ])
    def test_ratio_phrases(self, phrase, want):
        from src.scorer import categories_for_text
        assert categories_for_text(phrase) == want

    def test_forward_split_scores_as_split(self):
        from src.scorer import categories_for_items
        assert categories_for_items(['2-for-1 split']) == {'split'}
        assert categories_for_items(['1-for-3 reverse split']) == {'reverse_split'}


class TestStructuredF1:
    ALL_CODES = ['duplicate', 'missing', 'bad_price', 'split', 'reverse_split',
                 'unit_change', 'out_of_order', 'timezone', 'currency', 'adj_close',
                 'column_order', 'calendar']

    def test_pasting_full_list_fails_hard(self):
        scorer = TaskScorer(Path('tasks/answers'))
        score, _ = scorer.score_problems_found(self.ALL_CODES, ['unit_change_x', 'out_of_order_y'])
        assert score < 0.5

    def test_pasting_full_list_fails_easy(self):
        scorer = TaskScorer(Path('tasks/answers'))
        score, _ = scorer.score_problems_found(self.ALL_CODES, ['duplicate_row_x'])
        assert score < 0.5

    def test_exact_codes_score_one(self):
        scorer = TaskScorer(Path('tasks/answers'))
        score, _ = scorer.score_problems_found(
            ['unit_change', 'out_of_order'], ['unit_change_x', 'out_of_order_y'])
        assert score == 1.0

    def test_free_text_fallback_also_penalised(self):
        scorer = TaskScorer(Path('tasks/answers'))
        score, _ = scorer.score_problems_found(
            ['duplicate rows, missing days, bad spike, split, unit change, '
             'out of order rows, timezone issue, currency issue'],
            ['duplicate_row_x'])
        assert score < 0.5

    def test_dst_junk_does_not_pass(self, tmp_path):
        import json as _json
        answer = {'mean_return': 0.001, 'std_return': 0.02, 'annualized_volatility': 0.3,
                  'num_clean_days': 12, 'problems_found': ['timezone_x', 'out_of_order_y']}
        (tmp_path / 'task_h_1001_answer.json').write_text(_json.dumps(answer))
        scorer = TaskScorer(tmp_path)
        for bad_problems in (['junk'], []):
            result = scorer.score_task('task_h_1001', _json.dumps(
                {**{k: v for k, v in answer.items() if k != 'problems_found'},
                 'problems': bad_problems}))
            assert result['passed'] is False
            assert result['component_scores']['problems'] == 0.0


class TestGuessBaselines:
    """A fixed guess per task type, or pasting every code, must rarely pass.

    Passing requires exact problem-set identification (F1 = 1.0), so the
    guess pass rate is the fraction of tasks whose key set exactly equals
    the guessed set (with correct numbers assumed).
    """

    def test_guess_baselines_seeds_1_to_40(self):
        from collections import Counter, defaultdict

        from src.generate_tasks import TaskGenerator
        from src.hard_tier import HardTaskGenerator
        from src.scorer import categories_for_items

        hard_by_type = defaultdict(list)
        for seed in range(1, 41):
            gen = HardTaskGenerator(seed=seed)
            for maker in gen.task_generators():
                task, _payload = maker()
                hard_by_type[task['type']].append(
                    frozenset(categories_for_items(task['expected']['problems_found'])))

        all_codes = ['duplicate', 'missing', 'bad_price', 'split', 'reverse_split',
                     'unit_change', 'out_of_order', 'timezone', 'currency', 'adj_close',
                     'column_order', 'calendar']
        total, fixed_passes, paste_passes = 0, 0, 0
        for _type, sets in hard_by_type.items():
            most_common, count = Counter(sets).most_common(1)[0]
            assert count / len(sets) <= 0.20, (_type, most_common, count, len(sets))
            fixed_passes += count
            paste_passes += sum(1 for s in sets if s == frozenset(
                categories_for_items(all_codes)))
            total += len(sets)
        assert fixed_passes / total <= 0.20
        assert paste_passes / total <= 0.02

        easy_sets = []
        for seed in range(1, 41):
            gen = TaskGenerator(seed=seed)
            for maker in (gen.generate_task_return_calculation, gen.generate_task_backtest,
                          gen.generate_task_volatility):
                task, _df = maker()
                easy_sets.append(frozenset(categories_for_items(task['expected']['problems_found'])))
        easy_paste = sum(1 for s in easy_sets
                         if s == frozenset(categories_for_items(
                             ['duplicate', 'missing', 'bad_price', 'split'])))
        assert easy_paste / len(easy_sets) <= 0.02


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

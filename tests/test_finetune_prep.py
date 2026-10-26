"""Tests for fine-tuning data preparation."""

import json

from src.finetune_prep import FineTunePrep


def test_target_uses_problems_codes():
    prep = FineTunePrep()
    task = {
        'prompt': 'p', 'data': 'd',
        'expected': {'problems_found': ['duplicate_row_2024-01-15',
                                        'bad_price_2024-02-01_10.0'],
                     'mean_return': 0.001, 'std_return': 0.02},
    }
    example = prep.format_for_training(task)
    content = json.loads(example['messages'][2]['content'])
    assert content['problems'] == ['bad_price', 'duplicate']
    assert 'problems_found' not in content
    assert content['mean_return'] == 0.001

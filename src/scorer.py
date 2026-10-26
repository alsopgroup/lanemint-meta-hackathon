"""
Deterministic scorer for benchmark tasks.

Problem identification is scored by category. Free-text answers are matched
with synonym patterns; every occurrence is checked for negation in its own
clause (leading or trailing), uncertainty phrases such as "cannot rule out"
do not count, and word boundaries prevent e.g. "nonduplicated" matching.
Answers may also use the category codes directly.
"""

import json
import re
from pathlib import Path

_CLAUSE_SEP = ";.,()"
_NEG_CUES = {
    "no", "not", "none", "nothing", "never", "without", "cannot", "can't",
    "couldn't", "wasn't", "isn't", "aren't", "doesn't", "don't", "won't", "zero",
}
_UNCERTAIN = re.compile(r"\b(cannot|can't|can not|could not|couldn't) rule out\b")

# category -> regex on text with underscores turned into spaces
_PATTERNS = {
    "duplicate": r"\bduplicat\w*\b",
    "missing": r"\bmissing\b|\bgaps?\b",
    "bad_price": r"\bbad(?:ly)?\b|\bincorrect\w*\b|\bspikes?\b|\bjump\w*\b",
    "split": r"\bsplits?\b",
    "reverse_split": r"\breverse splits?\b|\b1-for-\d+\b|\b\d+-for-1\b",
    "unit_change": r"\bunit change\b|\bdollars? to cents\b|\bcents\b|\b100x\b",
    "out_of_order": r"\bout of (?:chronological )?order\b|\bchronological order\b"
                    r"|\bunsorted\b|\bout of sequence\b|\brow order\b",
    "timezone": r"\btime ?zones?\b|\btimezone\b|\bdst\b|\bdaylight\b|\bmixed offsets?\b"
                r"|\bduplicate (?:instant|timestamp|hour)\b",
    "currency": r"\bcurrency\b|\bcurrencies\b|\bfx\b|\bforeign exchange\b"
                r"|\bquoted in (?:eur|euros|usd|dollars)\b",
    "adj_close": r"\badj close\b|\badjusted close\b",
    "column_order": r"\bcolumn order\b|\bheader\b|\bcolumns? (?:reordered|changed|swapped)\b",
    "calendar": r"\bcalendar\b|\bmisalign\w*\b|\bholidays?\b|\btrading calendars?\b",
}


def _clause(text, start, end):
    lo = max(text.rfind(sep, 0, start) for sep in _CLAUSE_SEP) + 1
    stops = [text.find(sep, end) for sep in _CLAUSE_SEP]
    stops = [s for s in stops if s != -1]
    hi = min(stops) if stops else len(text)
    return text[lo:hi]


def _occurrence_positive(text, match):
    clause = _clause(text, match.start(), match.end())
    if _UNCERTAIN.search(clause):
        return False
    cues = [(m.group(0), m.start(), m.end())
            for m in re.finditer(r"\b\w+(?:'t)?\b", clause)]
    clause_lo = max(text.rfind(sep, 0, match.start()) for sep in _CLAUSE_SEP) + 1
    m_lo, m_hi = match.start() - clause_lo, match.end() - clause_lo
    for word, s, e in cues:
        if word in _NEG_CUES or word.endswith("n't"):
            if e <= m_lo or s >= m_hi:  # leading or trailing negation
                return False
    return True


def categories_for_text(text):
    """Return the set of problem categories positively asserted in one string."""
    norm = str(text).lower().replace("_", " ")
    found = set()
    for category, pattern in _PATTERNS.items():
        for match in re.finditer(pattern, norm):
            if _occurrence_positive(norm, match):
                found.add(category)
                break
    if "unit_change" in found:
        found.add("bad_price")
    if "reverse_split" in found:
        found.add("split")
    return found


class TaskScorer:
    """Scores task responses with detailed feedback."""

    def __init__(self, answers_dir: Path):
        self.answers_dir = Path(answers_dir)

    def load_expected_answer(self, task_id: str) -> dict:
        answer_path = self.answers_dir / f"{task_id}_answer.json"
        with open(answer_path) as f:
            return json.load(f)

    def parse_response(self, response: str) -> dict:
        if '```json' in response:
            start = response.find('```json') + 7
            end = response.find('```', start)
            response = response[start:end].strip()
        elif '```' in response:
            start = response.find('```') + 3
            end = response.find('```', start)
            response = response[start:end].strip()
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError as e:
            return {'error': f'Failed to parse JSON: {e}'}

    def compare_float(self, actual: float, expected: float, tolerance: float = 1e-5) -> bool:
        return abs(actual - expected) < tolerance

    def score_problems_found(self, actual, expected) -> tuple[float, str]:
        if not actual and not expected:
            return 1.0, "No problems to identify"
        if not actual:
            return 0.0, "No problems identified"
        found_issues = set()
        for item in actual:
            found_issues |= categories_for_text(item)
        expected_issues = set()
        for item in expected:
            expected_issues |= categories_for_text(item)
        if found_issues == expected_issues:
            return 1.0, "All issue types correctly identified"
        missing = expected_issues - found_issues
        extra = found_issues - expected_issues
        score = len(found_issues & expected_issues) / max(len(expected_issues), 1)
        feedback = []
        if missing:
            feedback.append(f"Missed: {sorted(missing)}")
        if extra:
            feedback.append(f"False positives: {sorted(extra)}")
        return score, "; ".join(feedback) if feedback else "Partial match"

    def score_task(self, task_id: str, response: str) -> dict:
        expected = self.load_expected_answer(task_id)
        parsed = self.parse_response(response)
        if 'error' in parsed:
            return {'task_id': task_id, 'passed': False, 'score': 0.0,
                    'feedback': parsed['error']}
        scores, feedback = {}, []
        if 'problems_found' in expected:
            prob_score, prob_feedback = self.score_problems_found(
                parsed.get('problems_found', []), expected['problems_found'])
            scores['problems'] = prob_score
            if prob_score < 1.0:
                feedback.append(f"Problems: {prob_feedback}")

        numeric_fields = [
            ('mean_return', 1e-4), ('std_return', 1e-4), ('total_return', 1e-4),
            ('final_price', 0.01), ('annualized_volatility', 1e-4),
            ('max_drawdown', 1e-4), ('portfolio_return', 1e-4),
            ('mean_beta', 1e-4), ('beta', 1e-4),
        ]
        for field, tolerance in numeric_fields:
            if field in expected:
                actual_val = parsed.get(field)
                if actual_val is None:
                    scores[field] = 0.0
                    feedback.append(f"{field}: missing")
                elif not isinstance(actual_val, (int, float)) or isinstance(actual_val, bool):
                    scores[field] = 0.0
                    feedback.append(f"{field}: not a number")
                elif self.compare_float(actual_val, expected[field], tolerance):
                    scores[field] = 1.0
                else:
                    scores[field] = 0.0
                    feedback.append(f"{field}: expected {expected[field]}, got {actual_val}")

        for field in ('num_clean_days', 'num_assets', 'num_aligned_days', 'num_return_days'):
            if field in expected:
                if parsed.get(field) == expected[field]:
                    scores[field] = 1.0
                else:
                    scores[field] = 0.0
                    feedback.append(f"{field}: expected {expected[field]}, got {parsed.get(field)}")

        overall = sum(scores.values()) / len(scores) if scores else 0.0
        if not scores:
            feedback.append("No scoreable fields found")
        return {'task_id': task_id, 'passed': overall >= 0.8,
                'score': round(overall, 3), 'component_scores': scores,
                'feedback': '; '.join(feedback) if feedback else 'All checks passed'}

    def score_all(self, results_file: Path) -> dict:
        with open(results_file) as f:
            results = json.load(f)
        scores = [self.score_task(r['task_id'], r['response']) for r in results]
        num_passed = sum(1 for s in scores if s['passed'])
        avg = sum(s['score'] for s in scores) / len(scores) if scores else 0
        return {'total_tasks': len(scores), 'passed': num_passed,
                'failed': len(scores) - num_passed,
                'pass_rate': round(num_passed / len(scores), 3) if scores else 0,
                'average_score': round(avg, 3), 'scores': scores}


def main():
    """CLI for scoring results."""
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m src.scorer <results_file>")
        print("Example: python -m src.scorer output/results.json")
        sys.exit(1)
    results_file = Path(sys.argv[1])
    answers_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('tasks/answers')
    if not results_file.exists():
        print(f"Error: Results file not found: {results_file}")
        sys.exit(1)
    if not answers_dir.exists():
        print(f"Error: Answers directory not found: {answers_dir}")
        print("Run: python -m src.generate_tasks first")
        sys.exit(1)
    scorer = TaskScorer(answers_dir)
    summary = scorer.score_all(results_file)
    print("\n=== Scoring Summary ===")
    print(f"Total tasks: {summary['total_tasks']}")
    print(f"Passed: {summary['passed']}")
    print(f"Failed: {summary['failed']}")
    print(f"Pass rate: {summary['pass_rate']:.1%}")
    print(f"Average score: {summary['average_score']:.3f}")
    print("\n=== Detailed Results ===")
    for score in summary['scores']:
        status = "✓ PASS" if score['passed'] else "✗ FAIL"
        print(f"{score['task_id']}: {status} (score: {score['score']:.3f})")
        if score['feedback'] != 'All checks passed':
            print(f"  {score['feedback']}")
    output_file = results_file.parent / 'scoring_summary.json'
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nDetailed results saved to: {output_file}")


if __name__ == '__main__':
    main()

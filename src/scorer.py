"""
Deterministic scorer for benchmark tasks.

Problem identification uses structured output: prompts ask for a JSON
`problems` array whose entries are category codes from a fixed list, and it
is scored with precision/recall (F1), so listing extra codes is penalised.
A legacy free-text `problems_found` array is still accepted as a fallback
and is mapped to categories with the same F1 scoring; that mapping is a
best-effort synonym/negation heuristic and is not exhaustive.
"""

import json
import re
from pathlib import Path

_CLAUSE_SEP = ";.,()"
_NEG_CUES = {
    "no", "not", "none", "nothing", "never", "without", "cannot", "can't",
    "couldn't", "wasn't", "isn't", "aren't", "doesn't", "don't", "won't", "zero",
    "neither", "nor", "lacks", "lack", "absence", "free",
}
_UNCERTAIN = re.compile(r"\b(cannot|can't|can not|could not|couldn't) rule out\b")
_IDIOM_NEG = re.compile(
    r"\bfree of\b|\bruled out\b|\babsence of\b|\blacks?\b|\bneither\b")
_UNCERTAIN_ADJ = {"possible", "possibly", "potential", "potentially",
                  "suspected", "unconfirmed", "maybe", "perhaps", "unlikely"}
# A post-separator detail negates its code only when it is *purely* a
# none/zero/no-count phrase ("none", "0", "0 rows", "not found",
# "nothing missing"). A detail that goes on to describe a finding
# ("no row for 2024-05-27", "not unique on 2024-03-04") never negates,
# and neither does a decimal (0.1, 0,1 or 0x10 are values, not counts).
_NOTHING_DETAIL = re.compile(
    r"^\s*(?:none|nothing|absent|zero|no|not|n/?a|not found|not detected|"
    r"not present|no issues?|ruled out)"
    r"(?:\s+(?:found|detected|present|missing|issues?|rows?|days?|bars?|entries))?"
    r"\s*[.!]?\s*$")
_ZERO_COUNT_DETAIL = re.compile(
    r"^\s*0(?!\.?\d)(?![.,]\d)(?!\w)"
    r"(?:\s+(?:rows?|days?|found|issues?|detected|present|bars?|entries))?"
    r"\s*[.!]?\s*$")


def _is_nothing_detail(detail):
    """True when a post-separator detail is purely a 'nothing found' phrase."""
    if re.search(r"\d{4}-\d{2}-\d{2}|\d{1,2}:\d{2}", detail):
        return False  # a named date or timestamp is a finding, not a negation
    return bool(_NOTHING_DETAIL.match(detail) or _ZERO_COUNT_DETAIL.match(detail))
CANONICAL_CATEGORIES = {
    "duplicate", "missing", "bad_price", "split", "reverse_split", "unit_change",
    "out_of_order", "timezone", "currency", "adj_close", "column_order", "calendar",
}

# category -> regex on text with underscores turned into spaces
_PATTERNS = {
    "duplicate": r"\bduplicat\w*\b",
    "missing": r"\bmissing\b|\bgaps?\b",
    "bad_price": r"\bbad(?:ly)?\b|\bincorrect\w*\b|\bspikes?\b|\bjump\w*\b",
    "split": r"\bsplits?\b",
    "reverse_split": r"\breverse splits?\b",
    "unit_change": r"\bunit change\b|\bdollars? to cents\b|\bcents\b|\b100x\b",
    "out_of_order": r"\bout of (?:chronological )?order\b|\bchronological order\b"
                    r"|\bunsorted\b|\bout of sequence\b|\brow order\b",
    "timezone": r"\btime ?zones?\b|\btimezone\b|\bdst\b|\bdaylight\b|\bmixed offsets?\b"
                r"|\btimestamps? mix\w*\b|\bduplicate (?:instant|timestamp|hour)\b",
    "currency": r"\bcurrency\b|\bcurrencies\b|\bfx\b|\bforeign exchange\b"
                r"|\bquoted in (?:eur|euros|usd|dollars)\b",
    "adj_close": r"\badj close\b|\badjusted close\b",
    "column_order": r"\bcolumn order\b|\bheader\b|\bcolumns? (?:reordered|changed|swapped)\b",
    "calendar": r"\bcalendar\b|\bmisalign\w*\b|\bholidays?\b|\btrading calendars?\b",
}


_CLAUSE_WORD_SEPS = (" - ", " — ", " but ", " although ", " however ")


def _clause_bounds(text, start, end):
    lo = max(text.rfind(sep, 0, start) for sep in _CLAUSE_SEP) + 1
    for dash in _CLAUSE_WORD_SEPS:
        idx = text.rfind(dash, 0, start)
        if idx != -1:
            lo = max(lo, idx + len(dash))
    stops = [text.find(sep, end) for sep in _CLAUSE_SEP]
    stops = [s for s in stops if s != -1]
    for dash in _CLAUSE_WORD_SEPS:
        idx = text.find(dash, end)
        if idx != -1:
            stops.append(idx)
    hi = min(stops) if stops else len(text)
    return lo, hi


def _clause(text, start, end):
    lo, hi = _clause_bounds(text, start, end)
    return text[lo:hi]


def _occurrence_positive(text, match):
    # "checked for duplicates but found none": the trailing clause negates
    # this match unless it introduces a different keyword of its own.
    tail = text[match.end():]
    stop = tail.find(".")
    segment = tail[:stop] if stop != -1 else tail
    m = re.search(r"\bbut (?:found )?(none|no|not)\b", segment)
    if m and not re.search(r"\b(duplicate|missing|gap|split|bad|spike)s?\b",
                           segment[m.end():]):
        return False
    clause = _clause(text, match.start(), match.end())
    # An explicit count of 0 immediately before the keyword ("0 duplicates
    # found", "there are 0 duplicate rows") means not detected. Decimals
    # such as 0.1 do not match: the 0 must be a standalone token.
    lo_bound, _ = _clause_bounds(text, match.start(), match.end())
    if re.search(r"(?:^|\s)0(?!\.?\d)\s+(?:\w+\s+)?$",
                 text[lo_bound:match.start()]):
        return False
    if _UNCERTAIN.search(clause) or _IDIOM_NEG.search(clause):
        # "besides/except" re-asserts what follows it: "no issues besides a duplicate"
        if not re.search(r"\b(besides|except)\b", clause):
            return False
    if text[match.end():match.end() + 5].startswith("-free"):
        return False
    lo, _hi = _clause_bounds(text, match.start(), match.end())
    m_lo, m_hi = match.start() - lo, match.end() - lo
    # an exception marker between a leading negation and the match re-asserts it
    exception_at = None
    exc = re.search(r"\b(besides|except)\b", clause[:m_lo])
    if exc:
        exception_at = exc.end()
    # Numeric tokens (e.g. the 0.1 in a bad-price label or a date) are
    # never negation cues; only words can negate.
    cues = [(m.group(0), m.start(), m.end())
            for m in re.finditer(r"\b\w+(?:'t)?\b", clause)
            if not any(ch.isdigit() for ch in m.group(0))]
    for word, s, e in cues:
        if word in _UNCERTAIN_ADJ and e <= m_lo:
            return False
        if word in _NEG_CUES or word.endswith("n't"):
            if e <= m_lo:
                if exception_at is not None and e <= exception_at:
                    continue
                return False
            if s >= m_hi:  # trailing negation
                return False
    return True


def _canonical(found, text=""):
    found = set(found)
    if "reverse_split" in found:
        found.discard("split")
    if "unit_change" in found:
        found.discard("bad_price")
    if "timezone" in found and "duplicate" in found and \
            any(w in text for w in ("instant", "timestamp", "hour")):
        found.discard("duplicate")
    return found


def categories_for_text(text):
    """Return the set of problem categories positively asserted in one string."""
    norm = str(text).lower().replace("_", " ")
    found = set()
    for category, pattern in _PATTERNS.items():
        for match in re.finditer(pattern, norm):
            if _occurrence_positive(norm, match):
                found.add(category)
                break
    # Split ratios are read numerically, per the prompt convention
    # "A:B (or A-for-B) = A new shares for B old shares": A>B is a
    # forward split, A<B a reverse split (so 1:0.5 is a forward split).
    for match in re.finditer(r"\b(\d+(?:\.\d+)?)\s*(?::|-for-)\s*(\d+(?:\.\d+)?)\b",
                             norm):
        new_shares, old_shares = float(match.group(1)), float(match.group(2))
        if old_shares > 0 and _occurrence_positive(norm, match):
            if new_shares / old_shares > 1:
                found.add("split")
            elif new_shares / old_shares < 1:
                found.add("reverse_split")
    return _canonical(found, norm)


_CODES_BY_LENGTH = sorted(CANONICAL_CATEGORIES, key=len, reverse=True)


def categories_for_key_items(items):
    """Categories from answer-key problem strings.

    Key strings are machine-generated labels such as
    ``bad_price_2024-07-09_0.1`` or ``calendar_misalignment_US_EU_holidays``.
    They are parsed by code prefix only: no negation or free-text
    heuristics apply, so numeric suffixes can never suppress a category.
    """
    found = set()
    for item in items or []:
        text = str(item).strip().lower()
        if text.startswith("stock_split"):
            found.add("split")
            continue
        for code in _CODES_BY_LENGTH:
            if text == code or text.startswith((code + "_", code + ":", code + " ")):
                found.add(code)
                break
        else:
            found |= categories_for_items([item])
    return _canonical(found)


def _head_code(segment):
    """Resolve the code named by the head of a 'head: detail' answer, if any."""
    seg = segment.strip().lower().replace(" ", "_")
    for code in _CODES_BY_LENGTH:
        if seg.startswith(code):
            return code
    if seg.endswith("s") and seg[:-1] in CANONICAL_CATEGORIES:
        return seg[:-1]
    if "duplicat" in seg:
        return "duplicate"
    if "bad" in seg:
        return "bad_price"
    if "missing" in seg or "gap" in seg:
        return "missing"
    if "split" in seg:
        return "split"
    return None


def categories_for_items(items):
    """Categories from a problems list.

    An entry that starts with a canonical code (``code`` or ``code: detail``)
    is a structured assertion of exactly that code; anything else falls back
    to free-text matching.
    """
    found = set()
    for item in items or []:
        text = str(item).strip().lower()
        # Explicit non-detections: a code-like head, a separator, and a
        # detail that is purely a nothing/zero phrase ("duplicates: 0",
        # "missing days: 0", "split - none", "count of duplicates: 0").
        # A detail describing a finding never negates (see _is_nothing_detail).
        neg = re.match(r"^(.+?)\s*(?::|—|–|-)\s*(.*)$", text)
        if neg and _head_code(neg.group(1)) is not None \
                and _is_nothing_detail(neg.group(2)):
            continue
        m = re.match(r"^([a-z_]+)\s*(?::|—|–|$)", text)
        code = m.group(1).replace(" ", "_") if m else None
        if code in CANONICAL_CATEGORIES:
            detail = text[m.end():]
            if not _is_nothing_detail(detail):
                found.add(code)
            continue
        # A verbatim key string (``bad_price_2024-07-09_0.1``) is an
        # assertion of its leading code, suffix and all; a purely-nothing
        # detail after a separator (``duplicate_rows: none``) is not.
        key_code = next((c for c in _CODES_BY_LENGTH
                         if text.startswith(c + "_")), None)
        if key_code is not None:
            rest = text[len(key_code):]
            sep = re.search(r":", rest) or re.search(r"[—–-]", rest)
            if not (sep and _is_nothing_detail(rest[sep.end():])):
                found.add(key_code)
        else:
            found |= categories_for_text(text)
    return _canonical(found)


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
        """Score category identification with precision/recall (F1).

        Extra categories lower precision, so listing every possible code
        scores poorly on both tiers.
        """
        expected_issues = categories_for_key_items(expected)
        if not expected_issues:
            return (1.0, "No problems to identify") if not categories_for_items(actual) \
                else (0.0, "False positives: problems reported where none expected")
        if not actual:
            return 0.0, "No problems identified"
        found_issues = categories_for_items(actual)
        tp = len(found_issues & expected_issues)
        if tp == 0:
            score = 0.0
        else:
            precision = tp / len(found_issues)
            recall = tp / len(expected_issues)
            score = 2 * precision * recall / (precision + recall)
        if found_issues == expected_issues:
            return 1.0, "All issue types correctly identified"
        feedback = []
        if expected_issues - found_issues:
            feedback.append(f"Missed: {sorted(expected_issues - found_issues)}")
        if found_issues - expected_issues:
            feedback.append(f"False positives: {sorted(found_issues - expected_issues)}")
        return score, "; ".join(feedback) if feedback else "Partial match"

    def score_task(self, task_id: str, response: str) -> dict:
        expected = self.load_expected_answer(task_id)
        parsed = self.parse_response(response)
        if 'error' in parsed:
            return {'task_id': task_id, 'passed': False, 'score': 0.0,
                    'feedback': parsed['error']}
        scores, feedback = {}, []
        if 'problems_found' in expected:
            actual_problems = parsed.get('problems', parsed.get('problems_found', []))
            prob_score, prob_feedback = self.score_problems_found(
                actual_problems, expected['problems_found'])
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
        # Passing requires exact problem-set identification (F1 = 1.0):
        # correct numbers plus a guessed or partial category set must not
        # pass, because the bundles are randomised enough that a fixed
        # per-type guess matches only a small minority of tasks (measured
        # in tests/test_scorer.py::TestGuessBaselines).
        problems_ok = scores.get('problems', 1.0) >= 1.0
        return {'task_id': task_id, 'passed': overall >= 0.8 and problems_ok,
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

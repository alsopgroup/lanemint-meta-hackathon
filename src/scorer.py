"""
Deterministic scorer for benchmark tasks.

Problem identification uses structured output: prompts ask for a JSON
`problems` array whose entries are category codes from a fixed list, and it
is scored with precision/recall (F1), so listing extra codes is penalised.
A legacy free-text `problems_found` array is still accepted as a fallback
and is mapped to categories with the same F1 scoring; that mapping is a
best-effort synonym/negation heuristic and is not exhaustive. Note: when
a split code is followed by a colon, the ratio after the colon decides
the category ("split: 1:2" is a reverse split, "split: 2:1" a split),
and a bare A:B ratio is only read as a share ratio when it cannot be a
clock time (see categories_for_text).
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
    "source_conflict",
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
                r"|\bmixed\b[^.;]{0,40}\boffsets?\b|\btimestamps? mix\w*\b"
                r"|\bduplicate (?:instant|timestamp|hour)\b",
    "currency": r"\bcurrency\b|\bcurrencies\b|\bfx\b|\bforeign exchange\b"
                r"|\bquoted in (?:eur|euros|usd|dollars)\b",
    "adj_close": r"\badj close\b|\badjusted close\b",
    "column_order": r"\bcolumn order\b|\bheader\b|\bcolumns? (?:reordered|changed|swapped)\b",
    "calendar": r"\bcalendar\b|\bmisalign\w*\b|\bholidays?\b|\btrading calendars?\b",
    "source_conflict": r"\bsource conflicts?\b|\bvendor (?:conflict|disagreement)\w*\b",
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
    if "timezone" in found and "duplicate" in found:
        if re.search(r"\bduplicate (?:instant|timestamp|hour)\s+\d{4}-\d{2}-\d{2}",
                     text):
            # A duplicated row at a named timestamp is a duplicate; only a
            # generic repeated instant (no concrete time) is timezone-only.
            found.discard("timezone")
        elif any(w in text for w in ("instant", "timestamp", "hour")):
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
    # A bare A:B is only read as a ratio when it cannot be a clock time:
    # an A-for-B form always counts; a colon form counts when "split" or
    # "ratio" appears in the same clause, or when B is not exactly two
    # digits (3:1, 10:1, 1:0.5 count; 09:30 does not). Colon ratios of
    # 50:1 or more are unit scales, not share splits.
    for match in re.finditer(r"\b(\d+(?:\.\d+)?)\s*(?::|-for-)\s*(\d+(?:\.\d+)?)\b",
                             norm):
        new_shares, old_shares = float(match.group(1)), float(match.group(2))
        if old_shares <= 0:
            continue
        quotient = new_shares / old_shares
        if "-for-" not in match.group(0):
            clause = _clause(norm, match.start(), match.end())
            two_digit_b = re.fullmatch(r"\d{2}", match.group(2)) is not None
            if two_digit_b and not re.search(r"\bsplits?\b|\bratios?\b", clause):
                continue  # a clock time, not a ratio
            if not 0.02 < quotient < 50:
                continue  # a unit scale (e.g. 100:1 to cents), not a split
        if _occurrence_positive(norm, match):
            if quotient > 1:
                found.add("split")
            elif quotient < 1:
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
    explicit = set()
    for item in items or []:
        text = str(item).strip().lower()
        if text.startswith("stock_split"):
            found.add("split")
            explicit.add("split")
            continue
        for code in _CODES_BY_LENGTH:
            if text == code or text.startswith((code + "_", code + ":", code + " ")):
                found.add(code)
                explicit.add(code)
                break
        else:
            found |= categories_for_items([item])
    found = _canonical(found)
    if {"split", "reverse_split"} <= explicit:
        # A key may explicitly assert both (e.g. a split table with
        # forward and reverse events); the free-text "reverse wins"
        # canonicalisation must not erase an explicit assertion.
        found |= {"split", "reverse_split"}
    return found


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
    to free-text matching. Exception: ``split: <ratio>`` reads the ratio
    after the colon, so ``split: 1:2`` asserts ``reverse_split``.
    """
    found = set()
    explicit = set()
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
                if code == "split":
                    # "split: <ratio>" reads the ratio after the colon:
                    # a ratio below 1 is a reverse split, not a split.
                    ratio = re.search(
                        r"(\d+(?:\.\d+)?)\s*(?::|-for-)\s*(\d+(?:\.\d+)?)", detail)
                    if ratio and float(ratio.group(2)) > 0 and \
                            float(ratio.group(1)) / float(ratio.group(2)) < 1:
                        found.add("reverse_split")
                        explicit.add("reverse_split")
                        continue
                found.add(code)
                explicit.add(code)
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
                explicit.add(key_code)
        else:
            found |= categories_for_text(text)
    found = _canonical(found)
    if {"split", "reverse_split"} <= explicit:
        # Both were explicitly asserted as structured entries (a split
        # table can contain forward and reverse events); keep both.
        found |= {"split", "reverse_split"}
    return found


def _placeholder_run(value: float) -> int:
    """Longest sequential run in a number's fractional digits
    (1.023456 -> 6, from 023456; integers use their whole digit string)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or value != value or value in (float('inf'), float('-inf')):
        return 0
    text = repr(abs(value))
    if 'e' in text:
        return 0
    # Fractional digits only: a run crossing the decimal point (the 1
    # and 0 in 1.023456) is an artefact, and correct answers such as
    # 0.212345 or 1.234588 get their apparent run that way.
    digits = text.split('.', 1)[1] if '.' in text else text
    if len(digits) < 5:
        return 0
    # Magnitudes below 0.1 (0.012345, 0.0456789) reach a 6-digit run
    # naturally when a correct small return rounds onto a sequential
    # pattern -- measured on the frontier reference answers (F2 seed
    # 8516: reconciled_total_return is exactly 0.012345). The typed
    # placeholders this hunts (0.123456, 1.023456, 0.987654) all sit at
    # magnitude >= 0.1 or carry a nonzero integer part, so runs below
    # 0.1 are never counted.
    if '.' in text and text.split('.', 1)[0] == '0' \
            and digits.startswith('0'):
        return 0
    best = run = 1
    direction = 0
    for prev, cur in zip(digits, digits[1:], strict=False):
        step = int(cur) - int(prev)
        if step == direction and step in (1, -1):
            run += 1
        elif step in (1, -1):
            direction, run = step, 2
        else:
            direction, run = 0, 1
        best = max(best, run)
    return best


def _placeholder_digits(value: float) -> bool:
    """True for a lone sequential-digit placeholder: a fractional digit
    run of >= 6 (1.023456, 0.987654, 0.654321). Genuine answers reach at
    most a 5-digit run (0.212345, 1.187654), so a 5-digit run only counts
    towards the >=2-placeholder-values rule in _is_no_attempt."""
    return _placeholder_run(value) >= 6


# Numeric fields that carry a family's headline computation; used only
# by the no_attempt analysis flag (never by scoring).
_CORE_NUMERIC_FIELDS = (
    'window_beta_1', 'window_beta_2', 'window_beta_3', 'window_beta_4',
    'last_window_beta', 'mean_beta', 'final_window_volatility',
    'reconciled_total_return', 'reconciled_max_drawdown',
    'beta_market', 'beta_sector', 'r_squared', 'final_return',
    'max_drawdown', 'first_rebalance_value')


def _field_tolerance(field: str, expected: dict) -> float:
    """Scoring tolerance for a float field (mirrors TaskScorer.score_task;
    the frontier max_drawdown case is keyed off first_rebalance_value
    in the expected answer instead of the task id)."""
    if field == 'final_price' or field.endswith('_price'):
        return 0.01
    if field == 'r_squared':
        return 5e-5
    if field in ('reconciled_max_drawdown', 'beta_market', 'beta_sector'):
        return 1e-3
    if field == 'max_drawdown' and 'first_rebalance_value' in expected:
        return 1e-3
    return 1e-4


def _is_no_attempt(parsed: dict, expected: dict | None = None) -> bool:
    """Tag placeholder-style answers: sequential-digit numbers, every
    numeric answer identical, or core fields left at exactly zero.
    Analysis only; the score itself is unchanged.

    Calibrated against the frontier reference answers (seeds 1-10000,
    10,000 correct answers per family, which produce zero flags): a
    single 5-digit fractional run is not enough, because correct
    answers contain values such as 0.212345 and 1.187654. The core-zero
    rules target answers that never attempted the headline computation:
    every core numeric field exactly 0, or at least two of them
    exactly 0 alongside a placeholder-style value (a >= 5-digit
    sequential run).

    When the expected answer is known (the scorer path), a numeric
    value that scores correct for its field is credited computation,
    not placeholder evidence, and is excluded from every rule below:
    a correct unrounded answer can itself carry a long sequential run
    (1.2345678 against a 1.234567 key)."""
    def credited(field, value):
        return expected is not None \
            and isinstance(expected.get(field), (int, float)) \
            and not isinstance(expected.get(field), bool) \
            and abs(value - expected[field]) \
            < _field_tolerance(field, expected)

    values = []

    def collect(node, field=None):
        if isinstance(node, dict):
            for k, v in node.items():
                collect(v, k)
        elif isinstance(node, list):
            for v in node:
                collect(v)
        elif isinstance(node, (int, float)) and not isinstance(node, bool):
            if field is None or not credited(field, node):
                values.append(node)

    collect({k: v for k, v in parsed.items()
             if k not in ('problems', 'problems_found')})
    runs = [_placeholder_run(v) for v in values]
    if any(r >= 6 for r in runs):
        return True
    if sum(1 for r in runs if r >= 5) >= 2:
        return True
    if len(values) >= 3 and len(set(values)) == 1:
        return True
    core_fields = [f for f in _CORE_NUMERIC_FIELDS
                   if isinstance(parsed.get(f), (int, float))
                   and not isinstance(parsed.get(f), bool)
                   and (expected is None or f in expected)]
    if core_fields:
        zeros = sum(1 for f in core_fields
                    if parsed[f] == 0 and not credited(f, parsed[f]))
        if zeros == len(core_fields):
            return True
        if zeros >= 2 and any(r >= 5 for r in runs):
            return True
    return False


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
            result = json.loads(response.strip())
        except json.JSONDecodeError as e:
            return {'error': f'Failed to parse JSON: {e}'}
        if not isinstance(result, dict):
            # A top-level list/string/number/null is a parse failure for
            # scoring purposes (one model's whole answer was a
            # one-element list): score 0 for this task, never crash.
            return {'error': 'Failed to parse JSON: top-level value is '
                             f'{type(result).__name__}, not an object'}
        return result

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
        # Tools-frontier schema (justified protected-file change: the
        # SCOPE.md build brief requires per-family core/secondary schemas
        # in this scorer).  A ``_tools_frontier`` entry in the answer key
        # is metadata, not a scored field: it names the core fields, the
        # relative/absolute float tolerances, and any per-field rtol
        # overrides.  Task pass for such tasks is all core fields correct
        # plus the exact problem set (where present); existing tiers are
        # completely unaffected.
        tf_schema = expected.get('_tools_frontier') \
            if isinstance(expected.get('_tools_frontier'), dict) else None

        def tf_tolerance(field, exp_value):
            rtol = (tf_schema.get('field_rtol') or {}).get(
                field, tf_schema.get('rtol', 1e-6))
            return max(float(tf_schema.get('atol', 1e-8)),
                       float(rtol) * abs(float(exp_value)))

        if 'problems_found' in expected:
            actual_problems = parsed.get('problems', parsed.get('problems_found', []))
            prob_score, prob_feedback = self.score_problems_found(
                actual_problems, expected['problems_found'])
            scores['problems'] = prob_score
            if prob_score < 1.0:
                feedback.append(f"Problems: {prob_feedback}")

        # Every remaining expected field is scored by type: dicts entry by
        # entry, integers and strings (dates, tickers) exactly, floats with
        # a tolerance of 0.01 for price fields, 1e-3 for
        # reconciled_max_drawdown (a path statistic accumulated over the
        # whole reconciled series; low-effort runs landed within ~3e-4 of
        # it while endpoint shortcuts are off by orders of magnitude more)
        # and for the F3 factor betas beta_market/beta_sector (pilot
        # near-misses landed 3e-4-1e-3 off the exact OLS values), 1e-3 for
        # frontier max_drawdown (same accumulated-path reasoning; the
        # hard tier's max_drawdown tasks keep 1e-4), 5e-5 for r_squared
        # (the simple-vs-log-returns discriminator: at 80 rows a
        # log-return regression passes 0.33% at this tolerance, measured
        # over seeds 1-200, while correct near-miss answers whose betas
        # are exact but R-squared lands ~3.5e-5 off now pass), and
        # 1e-4 for everything else (returns, other betas, volatility,
        # correlations, values).
        for field, exp_val in expected.items():
            if field == 'problems_found':
                continue
            if field == '_tools_frontier':
                continue
            if isinstance(exp_val, dict):
                actual_map = parsed.get(field)
                for key, sub in exp_val.items():
                    comp = f"{field}.{key}"
                    actual_val = actual_map.get(key) \
                        if isinstance(actual_map, dict) else None
                    dict_tol = tf_tolerance(field, sub) \
                        if tf_schema is not None else 1e-4
                    if isinstance(actual_val, (int, float)) \
                            and not isinstance(actual_val, bool) \
                            and self.compare_float(actual_val, sub, dict_tol):
                        scores[comp] = 1.0
                    else:
                        scores[comp] = 0.0
                        feedback.append(f"{comp}: expected {sub}, got {actual_val}")
            elif isinstance(exp_val, bool):
                continue
            elif isinstance(exp_val, int):
                if parsed.get(field) == exp_val:
                    scores[field] = 1.0
                else:
                    scores[field] = 0.0
                    feedback.append(
                        f"{field}: expected {exp_val}, got {parsed.get(field)}")
            elif isinstance(exp_val, float):
                tolerance = tf_tolerance(field, exp_val) \
                    if tf_schema is not None else \
                    0.01 if field == 'final_price' \
                    or field.endswith('_price') else \
                    5e-5 if field == 'r_squared' else \
                    1e-3 if field in ('reconciled_max_drawdown',
                                      'beta_market', 'beta_sector') or (
                        field == 'max_drawdown'
                        and str(task_id).startswith('task_f_')) else 1e-4
                actual_val = parsed.get(field)
                if actual_val is None:
                    scores[field] = 0.0
                    feedback.append(f"{field}: missing")
                elif not isinstance(actual_val, (int, float)) \
                        or isinstance(actual_val, bool):
                    scores[field] = 0.0
                    feedback.append(f"{field}: not a number")
                elif self.compare_float(actual_val, exp_val, tolerance) \
                        and not (tf_schema is not None
                                 and field in ('max_drawdown', 'nav_max_drawdown')
                                 and actual_val > 0):
                    # Tools-frontier memos state the drawdown sign
                    # convention explicitly: a negative fraction (0.0
                    # only when the path never falls). A positive value
                    # uses the opposite convention and never scores, even
                    # within tolerance of a near-zero key.
                    scores[field] = 1.0
                else:
                    scores[field] = 0.0
                    feedback.append(f"{field}: expected {exp_val}, got {actual_val}")
            elif isinstance(exp_val, str) or exp_val is None:
                if parsed.get(field) == exp_val:
                    scores[field] = 1.0
                else:
                    scores[field] = 0.0
                    feedback.append(
                        f"{field}: expected {exp_val}, got {parsed.get(field)}")

        overall = sum(scores.values()) / len(scores) if scores else 0.0
        if not scores:
            feedback.append("No scoreable fields found")
        # Passing requires exact problem-set identification (F1 = 1.0):
        # correct numbers plus a guessed or partial category set must not
        # pass, because the bundles are randomised enough that a fixed
        # per-type guess matches only a small minority of tasks (measured
        # in tests/test_scorer.py::TestGuessBaselines).
        problems_ok = scores.get('problems', 1.0) >= 1.0
        # Frontier families have a required core numeric output: the
        # headline computation must be right, not just the codes and
        # secondary fields (codes plus secondary fields alone could
        # otherwise reach 0.8 with the core value guessed as 0.0).
        core_fields = [f for f in (
            'reconciled_total_return', 'reconciled_max_drawdown',
            'final_return', 'beta_market', 'beta_sector', 'r_squared')
            if f in expected]
        # Frontier F1's cores are the window betas and the final-window
        # volatility (identified by window_beta_1; the hard tier's
        # unrelated beta tasks are unaffected). Only the fields the
        # answer key actually carries are required: keys written before
        # the fourth window was added have window_beta_1..3 only, and a
        # matching 3-window answer must still pass against them.
        if 'window_beta_1' in expected:
            core_fields += [f for f in (
                'window_beta_1', 'window_beta_2', 'window_beta_3',
                'window_beta_4', 'final_window_volatility')
                if f in expected]
        # Frontier F4's max_drawdown is core too (identified by the
        # accompanying first_rebalance_value; pilot passes had it wrong
        # while reaching 0.8 overall). The hard tier's standalone
        # max_drawdown tasks have no first_rebalance_value and keep
        # their existing pass rule.
        if 'max_drawdown' in expected and 'first_rebalance_value' in expected:
            core_fields.append('max_drawdown')
        core_failed = [f for f in core_fields
                       if scores.get(f, 0.0) < 1.0]
        core_ok = not core_failed
        if core_failed:
            feedback.append(
                f"Core field(s) not correct: {', '.join(core_failed)}")
        passed = overall >= 0.8 and problems_ok and core_ok
        if tf_schema is not None:
            # Tools-frontier pass rule (SCOPE.md section 4): every core
            # field correct (dict cores expand to their entries) and the
            # problem set exact where present.
            tf_failed = []
            for field in tf_schema.get('core') or []:
                if isinstance(expected.get(field), dict):
                    tf_failed += [f"{field}.{k}" for k in expected[field]
                                  if scores.get(f"{field}.{k}", 0.0) < 1.0]
                elif scores.get(field, 0.0) < 1.0:
                    tf_failed.append(field)
            if tf_failed:
                feedback.append(
                    f"Core field(s) not correct: {', '.join(tf_failed)}")
            passed = not tf_failed and problems_ok
        if not passed and not feedback:
            # A failure must never read as success: if no per-field or
            # core message explains it, say so explicitly rather than
            # falling through to 'All checks passed'.
            feedback.append(
                "Failed: score, problem-set or core-field requirement "
                "not met")
        result = {'task_id': task_id,
                  'passed': passed,
                  'score': round(overall, 3), 'component_scores': scores,
                  'feedback': '; '.join(feedback) if feedback else 'All checks passed',
                  'analysis_flags': ['no_attempt']
                  if _is_no_attempt(parsed, expected) else []}
        # Frontier F4 checkpoint score: partial progress along the pipeline
        # (aligned days, first rebalance value, final return, drawdown).
        # Reported only; the pass rule above is unchanged.
        if 'first_rebalance_value' in expected:
            ckpt = [f for f in ('num_aligned_days', 'first_rebalance_value',
                                'final_return', 'max_drawdown', 'num_rebalances')
                    if f in scores]
            if ckpt:
                result['checkpoint_score'] = round(
                    sum(scores[f] for f in ckpt) / len(ckpt), 3)
        return result

    def score_all(self, results_file: Path) -> dict:
        with open(results_file) as f:
            results = json.load(f)
        scores = [self.score_task(r['task_id'], r['response']) for r in results]
        num_passed = sum(1 for s in scores if s['passed'])
        avg = sum(s['score'] for s in scores) / len(scores) if scores else 0
        return {'total_tasks': len(scores), 'passed': num_passed,
                'failed': len(scores) - num_passed,
                'pass_rate': round(num_passed / len(scores), 3) if scores else 0,
                'average_score': round(avg, 3),
                'no_attempt_tasks': sum(
                    1 for s in scores
                    if 'no_attempt' in s.get('analysis_flags', [])),
                'scores': scores}


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

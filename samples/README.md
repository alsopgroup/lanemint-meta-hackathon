# Sample tasks for judges

Ready-made easy-tier tasks you can try without installing anything or
generating data yourself. Each CSV is synthetic market data with a few
known problems planted in it; each answer key says exactly what was
planted and what the cleaned-up numbers should be.

These files were generated with the existing generator, using seed
**31** — a non-reserved seed (the reserved evaluation seeds are 42,
1001 and 1002, and were not used here):

```bash
python -m src.generate_tasks --tier easy --seed 31 --count 3 --output-dir samples
```

A test (`tests/test_samples.py`) regenerates that output into a
temporary directory and checks `samples/` still matches it byte for
byte, so these files cannot silently drift from the generator. A
second test (`app/tests/test_sample_answers.py`) keeps the app's
server-side recognised-sample hashes and keys in sync with this
folder.

## What's in each file

| File | Task type | Planted problems, in plain words |
|---|---|---|
| `task_001.csv` | Return calculation (`return_calculation`) | One bad price: the close on 2024-03-15 is a one-day spike at 10× its neighbours, then it reverts. One duplicate row: 2024-04-10 appears twice. One missing day: the weekday 2024-03-22 has no row. |
| `task_002.csv` | Volatility (`volatility`) | One duplicate row: 2024-07-29 appears twice. One missing day: the weekday 2024-07-15 has no row. |
| `task_003.csv` | Backtest (`backtest`) | One unadjusted 2-for-1 split: from 2024-05-17 the prices drop to about half their previous level and stay there, with volume doubled. One bad price: the close on 2024-05-30 is a one-day spike at 10× its neighbours, then it reverts. |

Together the three samples cover all four easy-tier problem kinds:
duplicates, missing days (gaps), bad prices and an unadjusted split.

The matching answer keys live in `answers/`:

| Answer key | For | Expected numbers |
|---|---|---|
| `answers/task_001_answer.json` | `task_001.csv` | `mean_return`, `std_return` |
| `answers/task_002_answer.json` | `task_002.csv` | `annualized_volatility`, `num_clean_days` |
| `answers/task_003_answer.json` | `task_003.csv` | `total_return`, `final_price` |

`tasks.json` holds the full task wording for each CSV (the prompt a
model would be given, without the expected answer), exactly as the
generator emits it.

Every CSV uses the generator's columns —
`date,ticker,open,high,low,close,volume` — and is small (a few dozen
to about a hundred rows), so each one fits the upload limits of the
demo app at https://meta.lanemint.com: the app accepts up to 500 data
rows and 20 columns, and these files are well under that.

## Try one in the app

1. Open https://meta.lanemint.com.
2. Scroll to **Run a check**. Either click **Try a sample →** in the
   hero to run the app's built-in sample, or use **Upload CSV** /
   **Drop your own CSV** and pick one of the CSVs from this folder
   (for example `task_001.csv`).
3. Run the check and read the findings: the model should flag the
   problems listed for that file in the table above, in plain English.
4. Open the **Score** tab to see the full score against the matching
   answer key, just like the app's built-in seeded samples. You can
   also compare by hand against the matching file in `answers/`.

How that scoring works: when you upload a file, the app hashes its
contents (SHA-256, after stripping any byte-order mark and
normalising line endings) and checks it against the three public
samples in this folder, whose answer keys are kept server-side. An
exact match is recognised and scored. **Any other CSV — even one of
these files with a single value edited — gets findings only:** its
Score tab is greyed out and disabled, with a note explaining that
scoring needs an answer key and that only files matching this
folder's samples can be scored in the app. For those files, compare
against a key by hand or score a model's JSON answer locally with
`src.scorer`, as shown below.

## How to read the answer key

Each answer key is a small JSON file. Its `problems_found` list names
every planted problem as `<kind>_<date>` (a split also carries its
ratio, a bad price its multiplier):

- `duplicate_row_2024-04-10` — the row for that date appears twice.
- `missing_day_2024-03-22` — that weekday is absent from the CSV.
- `bad_price_2024-03-15_10.0` — that day's close was multiplied by 10
  for one day only, then prices revert.
- `stock_split_2024-05-17_2.0` — an unadjusted 2-for-1 split takes
  effect on that date: later prices are halved, volume doubled.

The other fields are the numbers a correct answer should produce
after cleaning (removing duplicates and bad prices, noting gaps, and
adjusting pre-split prices onto the post-split basis), as defined by
the task wording in `tasks.json`.

To score a model's answer locally, save its JSON answers as a results
file — a list of `{"task_id": ..., "response": ...}` entries, where
`response` is the model's JSON answer as a string — and run:

```bash
python -m src.scorer my_results.json samples/answers
```

The scorer checks the reported problem categories against
`problems_found` and the numeric fields against the expected values,
and prints a pass/fail per task. See `src/scorer.py` for the exact
tolerances.

## Generate your own

Any non-reserved seed gives you a fresh, never-seen set:

```bash
python -m src.generate_tasks --tier easy --seed 123 --count 3 --output-dir my_samples
```

Do not use the reserved evaluation seeds (42, 1001, 1002) for training
or tuning data — those are held out for the headline results. All
data here and from the generator is synthetic; there is no real
market data in these files.

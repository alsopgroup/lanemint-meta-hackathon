"""
Deterministic scorer for benchmark tasks.

Evaluates model responses against expected answers with detailed feedback.
"""

import json
from pathlib import Path


class TaskScorer:
    """Scores task responses with detailed feedback."""

    def __init__(self, answers_dir: Path):
        """Initialize scorer with answers directory."""
        self.answers_dir = Path(answers_dir)

    def load_expected_answer(self, task_id: str) -> dict:
        """Load expected answer for a task."""
        answer_path = self.answers_dir / f"{task_id}_answer.json"
        with open(answer_path) as f:
            return json.load(f)

    def parse_response(self, response: str) -> dict:
        """Parse JSON response from model."""
        # Try to extract JSON from markdown code blocks
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
        """Compare floats with tolerance."""
        return abs(actual - expected) < tolerance

    def score_problems_found(
        self,
        actual: list[str],
        expected: list[str]
    ) -> tuple[float, str]:
        """
        Score problem identification.

        We're lenient here - the model needs to identify issues but exact
        matching of problem strings is hard since they're synthetic identifiers.
        We check for keywords instead.
        """
        if not actual:
            return 0.0, "No problems identified"

        actual_lower = [p.lower() for p in actual]
        expected_lower = [p.lower() for p in expected]

        # Check for key issue types
        found_issues = set()
        if any('duplicate' in p for p in actual_lower):
            found_issues.add('duplicate')
        if any('missing' in p or 'gap' in p for p in actual_lower):
            found_issues.add('missing')
        if any('bad' in p or 'incorrect' in p or 'spike' in p or 'jump' in p for p in actual_lower):
            found_issues.add('bad_price')
        if any('split' in p for p in actual_lower):
            found_issues.add('split')

        expected_issues = set()
        if any('duplicate' in p for p in expected_lower):
            expected_issues.add('duplicate')
        if any('missing' in p for p in expected_lower):
            expected_issues.add('missing')
        if any('bad' in p for p in expected_lower):
            expected_issues.add('bad_price')
        if any('split' in p for p in expected_lower):
            expected_issues.add('split')

        if found_issues == expected_issues:
            return 1.0, "All issue types correctly identified"

        missing = expected_issues - found_issues
        extra = found_issues - expected_issues

        score = len(found_issues.intersection(expected_issues)) / max(len(expected_issues), 1)

        feedback = []
        if missing:
            feedback.append(f"Missed: {missing}")
        if extra:
            feedback.append(f"False positives: {extra}")

        return score, "; ".join(feedback) if feedback else "Partial match"

    def score_task(self, task_id: str, response: str) -> dict:
        """Score a single task response."""
        expected = self.load_expected_answer(task_id)
        parsed = self.parse_response(response)

        if 'error' in parsed:
            return {
                'task_id': task_id,
                'passed': False,
                'score': 0.0,
                'feedback': parsed['error']
            }

        scores = {}
        feedback = []

        # Score problems found
        if 'problems_found' in expected:
            actual_problems = parsed.get('problems_found', [])
            prob_score, prob_feedback = self.score_problems_found(
                actual_problems,
                expected['problems_found']
            )
            scores['problems'] = prob_score
            if prob_score < 1.0:
                feedback.append(f"Problems: {prob_feedback}")

        # Score numeric fields
        numeric_fields = [
            ('mean_return', 1e-4),
            ('std_return', 1e-4),
            ('total_return', 1e-4),
            ('final_price', 0.01),
            ('annualized_volatility', 1e-4),
        ]

        for field, tolerance in numeric_fields:
            if field in expected:
                actual_val = parsed.get(field)
                expected_val = expected[field]

                if actual_val is None:
                    scores[field] = 0.0
                    feedback.append(f"{field}: missing")
                elif not isinstance(actual_val, (int, float)):
                    scores[field] = 0.0
                    feedback.append(f"{field}: not a number")
                elif self.compare_float(actual_val, expected_val, tolerance):
                    scores[field] = 1.0
                else:
                    scores[field] = 0.0
                    feedback.append(
                        f"{field}: expected {expected_val}, got {actual_val}"
                    )

        # Score integer fields
        if 'num_clean_days' in expected:
            actual_val = parsed.get('num_clean_days')
            expected_val = expected['num_clean_days']

            if actual_val == expected_val:
                scores['num_clean_days'] = 1.0
            else:
                scores['num_clean_days'] = 0.0
                feedback.append(
                    f"num_clean_days: expected {expected_val}, got {actual_val}"
                )

        # Calculate overall score
        if not scores:
            overall_score = 0.0
            feedback.append("No scoreable fields found")
        else:
            overall_score = sum(scores.values()) / len(scores)

        passed = overall_score >= 0.8  # 80% threshold

        return {
            'task_id': task_id,
            'passed': passed,
            'score': round(overall_score, 3),
            'component_scores': scores,
            'feedback': '; '.join(feedback) if feedback else 'All checks passed'
        }

    def score_all(self, results_file: Path) -> dict:
        """Score all task results from a results file."""
        with open(results_file) as f:
            results = json.load(f)

        scores = []
        for result in results:
            task_id = result['task_id']
            response = result['response']
            score = self.score_task(task_id, response)
            scores.append(score)

        num_passed = sum(1 for s in scores if s['passed'])
        avg_score = sum(s['score'] for s in scores) / len(scores) if scores else 0

        summary = {
            'total_tasks': len(scores),
            'passed': num_passed,
            'failed': len(scores) - num_passed,
            'pass_rate': round(num_passed / len(scores), 3) if scores else 0,
            'average_score': round(avg_score, 3),
            'scores': scores
        }

        return summary


def main():
    """CLI for scoring results."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.scorer <results_file>")
        print("Example: python -m src.scorer output/results.json")
        sys.exit(1)

    results_file = Path(sys.argv[1])
    answers_dir = Path('tasks/answers')

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

    # Save detailed results
    output_file = results_file.parent / 'scoring_summary.json'
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nDetailed results saved to: {output_file}")


if __name__ == '__main__':
    main()

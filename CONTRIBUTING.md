# Contributing to Meta Hackathon Benchmark

Thank you for your interest in contributing! This project was built for Meta's AI Developer Hackathon, and we welcome improvements and bug fixes.

## Getting Started

### Prerequisites

- Python 3.11+
- Git

### Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/lanemint-meta-hackathon.git
   cd lanemint-meta-hackathon
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run tests to verify setup:
   ```bash
   pytest tests/ -v
   ```

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b your-feature-name
```

Use descriptive branch names:
- `fix/scorer-tolerance`
- `feature/new-task-type`
- `docs/update-readme`

### 2. Make Changes

- Write clear, concise code
- Follow existing code style
- Add tests for new functionality
- Update documentation as needed

### 3. Test Your Changes

```bash
# Run unit tests
pytest tests/ -v

# Run full pipeline dry-run
python -m src.generate_tasks
python -m src.runner --dry-run --limit 3
python -m src.scorer output/results.json
```

### 4. Commit and Push

```bash
git add .
git commit -m "Brief description of changes"
git push origin your-feature-name
```

Write clear commit messages:
- Use present tense ("Add feature" not "Added feature")
- Keep first line under 72 characters
- Add details in body if needed

### 5. Open a Pull Request

- Fill out the PR template
- Link related issues
- Wait for CI checks to pass
- Respond to review feedback

## Code Style

- Follow PEP 8 conventions
- Use type hints where appropriate
- Keep functions focused and testable
- Add docstrings to public functions

## Testing Guidelines

- Write tests for new features
- Maintain or improve test coverage
- Use descriptive test names
- Test edge cases and error conditions

## Documentation

- Update README.md for user-facing changes
- Add docstrings to new functions/classes
- Include code examples where helpful
- Keep documentation concise and clear

## What to Contribute

### Welcome Contributions

- 🐛 Bug fixes
- ✨ New task types
- 📊 Improved scoring logic
- 📝 Documentation improvements
- ✅ Additional tests
- 🎨 Code quality improvements

### Before Contributing Large Changes

For significant changes, please open an issue first to discuss:
- New features
- Architectural changes
- Breaking changes

## Code Review Process

1. Automated checks must pass (tests, linting)
2. At least one maintainer approval required
3. Address review feedback
4. Squash commits if requested

## Questions?

- Open an issue for questions
- Check existing issues and PRs first
- Be respectful and constructive

## Project Constraints

This is a **public repository** with specific constraints:

- ❌ No secrets or API keys in code
- ❌ No paid API calls without approval
- ❌ No proprietary data or code from other projects
- ❌ No Meta logos or affiliation claims
- ✅ Keep it simple and self-contained
- ✅ Maintain dry-run mode for testing

---

Thank you for contributing! 🎉

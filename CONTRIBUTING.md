# Contributing to physics-lab

Thanks for your interest! This is a personal educational project, but contributions — new experiments, physics corrections, portability fixes, or the first test suite — are genuinely welcome.

## Getting set up

```bash
# 1. Fork and clone
git clone https://github.com/<your-username>/physics-lab.git
cd physics-lab

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies (note: Pillow is required beyond requirements.txt)
pip install numpy matplotlib pygame moderngl pillow
# optional GPU acceleration for the 2D quantum experiments:
pip install cupy-cuda12x           # match your CUDA version
```

You will also need **`ffmpeg` on your `PATH`** for any `--save` command, and — for the Bell experiment specifically — **Linux with libEGL**. See the [README](README.md#requirements) for the full runtime requirements.

Sanity-check your setup with the lightest experiment:

```bash
python "mecanica quantica/01-schrodinger-1d/main.py"
```

## Repository layout

Each experiment is a standalone entrypoint (`main.py` with `if __name__ == "__main__"`), grouped into three themed directories plus one shared engine:

```
relatividade geral/       # general relativity (black hole)
relatividade restrita/    # special relativity + common/engine.py (shared)
mecanica quantica/        # quantum mechanics
```

- The shared OpenGL engine lives in `relatividade restrita/common/engine.py`. Special-relativity apps subclass `BaseApp`; the black hole imports it via `sys.path.insert`. Prefer reusing it over duplicating GL boilerplate.
- **Directory names contain spaces**, so quote paths in every command: `python "mecanica quantica/01-schrodinger-1d/main.py"`.
- Shaders sit under each experiment's `shaders/` folder and are loaded from disk at runtime — there is no build step.

## Pull request flow

1. **Fork** the repo and create a branch off `main`:
   ```bash
   git checkout -b feat/my-change
   ```
2. **Make your change.** Keep experiments self-contained; if you touch physics, note any approximations in the code and the README so the project stays honest about what it computes.
3. **Test it manually.** Run the affected experiment (both interactive and `--save` paths if relevant) and confirm it still renders. If you add a `pytest` suite, run it and include the command in your PR description.
4. **Commit** using [Conventional Commits](https://www.conventionalcommits.org/) with a scope, e.g. `fix(bell): correct CHSH measurement angles`.
5. **Push** and open a pull request against `main`, filling in the PR template.

### Conventional commit types

| Type | When to use it |
|------|----------------|
| `feat` | A new experiment, control, or capability |
| `fix` | A bug fix (rendering, physics, crash) |
| `docs` | README, comments, or documentation only |
| `refactor` | Code change that neither fixes a bug nor adds a feature |
| `perf` | A performance improvement (e.g. shader or solver speedups) |
| `style` | Formatting only, no behavior change |
| `test` | Adding or updating tests |
| `chore` | Tooling, dependencies, CI, housekeeping |

Use the experiment or area as the scope: `feat(blackhole:)`, `fix(schrodinger-1d:)`, `chore(deps:)`.

## Tests, formatting, and linting

- **Tests:** there is currently **no test suite**, so there is no test command to run yet. Adding one is high on the wish list — good first targets are norm conservation for the Split-Step solvers and `S → 2√2` for the Bell sampler. If you introduce `pytest`, wire it into CI and document it here.
- **Formatting / linting:** the project does not yet enforce a formatter, but new code should be clean and consistent with the surrounding style. If you'd like to run tools locally:
  ```bash
  pip install ruff
  ruff check .
  ruff format .
  ```

## Reporting bugs & requesting features

Use the issue templates:

- [Bug report](.github/ISSUE_TEMPLATE/bug_report.yml)
- [Feature request](.github/ISSUE_TEMPLATE/feature_request.yml)

For security issues, follow [SECURITY.md](SECURITY.md) instead of opening a public issue.

By contributing, you agree that your contributions are licensed under the [MIT License](LICENSE) and that you will uphold the project's [Code of Conduct](CODE_OF_CONDUCT.md).

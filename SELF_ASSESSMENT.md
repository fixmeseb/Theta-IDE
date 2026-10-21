# Project 2 — Self-Assessed Rubric

**Repo:** <https://github.com/fixmeseb/Theta-IDE>

Scale: **0** none · **1** a little · **2** somewhat · **3** a lot.
Every row carries a link as evidence. The tutor spot-checks five rows at the demo,
so each score below is one a reviewer can verify from the repository.

Figures measured on commit `4c6d6af`: 769 commits, 1354 test functions across 106
files, 29% of Python lines in tests. Coverage is measured by CI.

---

## Teamwork, visible in GitHub

| Criterion | Score | Evidence |
|---|---|---|
| Many commits (exemplar has 467) | **3** | [769 commits on main](https://github.com/fixmeseb/Theta-IDE/commits/main) — 65% above the exemplar |
| Commits come from different people | **3** | [Contributors](https://github.com/fixmeseb/Theta-IDE/graphs/contributors) — 6 identities, 5 people |
| Workload is spread over the whole team | **2** | [Contributors](https://github.com/fixmeseb/Theta-IDE/graphs/contributors). Four people contribute substantially, three of them within 25% of each other, but one leads by more than the "3x more productive" the rubric allows. Note `kc` and `ccast44` are one person (same GitHub user id), so the graph splits one contributor in two |
| Team members work across many parts, not one silo each | **3** | [Commit history](https://github.com/fixmeseb/Theta-IDE/commits/main) — three of the four work substantially across `frontend/`, `src/` and `tests/`; no one is confined to a single area |
| Short release cycles: frequent small commits, merged promptly | **3** | [25 pull requests, all merged or closed, none open](https://github.com/fixmeseb/Theta-IDE/pulls?q=is%3Apr) |
| Whole team uses the same tools: config files in repo, updated by many | **3** | [`pyproject.toml`](https://github.com/fixmeseb/Theta-IDE/blob/main/pyproject.toml) (`[tool.ruff]`, `[tool.mypy]`, `[tool.coverage]`), [`.pre-commit-config.yaml`](https://github.com/fixmeseb/Theta-IDE/blob/main/.pre-commit-config.yaml), [`ci.yml`](https://github.com/fixmeseb/Theta-IDE/blob/main/.github/workflows/ci.yml) |
| Whole team can run the system on their own machine | **3** | Every member runs the IDE on their own machine following [`INSTALL.md`](https://github.com/fixmeseb/Theta-IDE/blob/main/INSTALL.md); demonstrated live at the demo |

## Issues and discussion

| Criterion | Score | Evidence |
|---|---|---|
| Issue reports: there are many | **1** | [3 issues total](https://github.com/fixmeseb/Theta-IDE/issues?q=is%3Aissue) |
| Issues are being closed | **3** | [All 3 closed, none open](https://github.com/fixmeseb/Theta-IDE/issues?q=is%3Aissue+is%3Aclosed) |
| Issues are discussed before they are closed | **3** | All 3 carry a closing summary: [#11](https://github.com/fixmeseb/Theta-IDE/issues/11), [#12](https://github.com/fixmeseb/Theta-IDE/issues/12), [#13](https://github.com/fixmeseb/Theta-IDE/issues/13). The rubric allows for this directly — "if the talk happened in chat, add a summary to the issue" — and each records the cause, the fixing commit and the tests |
| A large share of issues relate to handling failing cases | **3** | All 3 are failing-test reports: [#11](https://github.com/fixmeseb/Theta-IDE/issues/11), [#12](https://github.com/fixmeseb/Theta-IDE/issues/12), [#13](https://github.com/fixmeseb/Theta-IDE/issues/13) — each opened on a failure and closed by a fix |
| Chat channel exists and is active; poster QR points at it | **3** | Discord is the team's working channel and is active daily; the poster QR points at it |

## Tests

| Criterion | Score | Evidence |
|---|---|---|
| Tests exist: 50 is good, 100 is better | **3** | [1354 test functions](https://github.com/fixmeseb/Theta-IDE/tree/main/tests) across 106 files, expected and failure cases both |
| Tests are a serious share of the code base (~30%) | **3** | [`tests/`](https://github.com/fixmeseb/Theta-IDE/tree/main/tests) — 24,091 of 81,952 Python lines are tests, **29%** |
| Tests run automatically on every commit | **3** | [CI badge](https://github.com/fixmeseb/Theta-IDE/actions/workflows/ci.yml) — lint, test and typecheck on push and PR |
| Coverage is measured; the badge shows the number | **3** | **70%** line coverage measured by `pytest-cov`, configured in [`pyproject.toml`](https://github.com/fixmeseb/Theta-IDE/blob/main/pyproject.toml) and run by [CI](https://github.com/fixmeseb/Theta-IDE/blob/main/.github/workflows/ci.yml); the badge shows that number |

## Docs

| Criterion | Score | Evidence |
|---|---|---|
| Docs are generated, and not ugly | **2** | [`documentation/content.py`](https://github.com/fixmeseb/Theta-IDE/blob/main/frontend/plugins/core/documentation/content.py) — 23 styled HTML articles rendered in an in-app documentation browser with search and navigation, plus Markdown guides. Not auto-generated from docstrings, which is what holds this short of 3 |
| Docs, the WHAT: per class/function descriptions | **2** | [`src/`](https://github.com/fixmeseb/Theta-IDE/tree/main/src) — 1,061 of 1,916 public classes and functions carry docstrings, **55%** |
| Docs, the HOW: mini-tutorials with worked examples | **3** | [21 in-app articles](https://github.com/fixmeseb/Theta-IDE/blob/main/frontend/plugins/core/documentation/content.py) (1490 lines), plus [`COMPONENT_PLUGIN_GUIDE.md`](https://github.com/fixmeseb/Theta-IDE/blob/main/docs/COMPONENT_PLUGIN_GUIDE.md) and [`WORKFLOW_GUIDE.md`](https://github.com/fixmeseb/Theta-IDE/blob/main/docs/WORKFLOW_GUIDE.md) |
| Docs, the WHY: the docs tell a story | **3** | [`documentation/content.py`](https://github.com/fixmeseb/Theta-IDE/blob/main/frontend/plugins/core/documentation/content.py) — the in-app docs open with "Welcome & Architecture Overview" and "System Architecture", which set out what the IDE is for and why it is built in layers, before any API detail |
| Teaser video (12–30 s) at the TOP of README | **3** | [Embedded directly under the badges](https://github.com/fixmeseb/Theta-IDE#readme) |
| Feature walk-through video (2–5 min), linked in README | **3** | Linked in the [README](https://github.com/fixmeseb/Theta-IDE#readme) beneath the teaser |

## Standard files and bling

| Criterion | Score | Evidence |
|---|---|---|
| README.md: all badges above, all working | **3** | [README](https://github.com/fixmeseb/Theta-IDE#readme) — 8 badges: build, tests, coverage, style, types, pre-commit, licence, Python. Every URL fetched and returning 200; none is dead |
| .gitignore | **3** | [`.gitignore`](https://github.com/fixmeseb/Theta-IDE/blob/main/.gitignore) |
| INSTALL.md | **3** | [`INSTALL.md`](https://github.com/fixmeseb/Theta-IDE/blob/main/INSTALL.md) |
| LICENSE.md | **3** | [`LICENSE.md`](https://github.com/fixmeseb/Theta-IDE/blob/main/LICENSE.md) — MIT |
| CODE-OF-CONDUCT.md | **3** | [`CODE-OF-CONDUCT.md`](https://github.com/fixmeseb/Theta-IDE/blob/main/CODE-OF-CONDUCT.md) |
| CONTRIBUTING.md | **3** | [`CONTRIBUTING.md`](https://github.com/fixmeseb/Theta-IDE/blob/main/CONTRIBUTING.md) |
| Style checker, formatter, syntax checker: configs + badges | **3** | Ruff and mypy configured in [`pyproject.toml`](https://github.com/fixmeseb/Theta-IDE/blob/main/pyproject.toml), enforced by [CI](https://github.com/fixmeseb/Theta-IDE/blob/main/.github/workflows/ci.yml), badged in the README |
| DOI badge from Zenodo | **3** | [10.5281/zenodo.23537094](https://doi.org/10.5281/zenodo.23537094) — the concept DOI, which resolves to the newest release rather than going stale. Badged in the README |
| THIRD_PARTY_LIBRARIES.md: every dependency with its licence | **3** | [`THIRD_PARTY_LIBRARIES.md`](https://github.com/fixmeseb/Theta-IDE/blob/main/THIRD_PARTY_LIBRARIES.md) — 38 dependencies, generated from installed metadata, and it surfaces that PyQt6 is GPL-3.0 in an MIT project |
| AI use is visible and disclosed | **3** | [`.agents/AGENTS.md`](https://github.com/fixmeseb/Theta-IDE/blob/main/.agents/AGENTS.md), 252 lines of agent instructions committed to the repo. Human checking is evidenced on [25 pull requests](https://github.com/fixmeseb/Theta-IDE/pulls?q=is%3Apr), several carrying line-by-line review that changed the code before merge |
| Repo holds no keys or passwords | **3** | Scanned for hardcoded credentials; none found. [CI](https://github.com/fixmeseb/Theta-IDE/blob/main/.github/workflows/ci.yml) keeps the one token in `secrets.CODECOV_TOKEN`, not in the repo |

---

## Subtotal

All **33 rows scored**.

**Total: 94 / 99** (94%).

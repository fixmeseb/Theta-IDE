# Project 2 — Self-Assessed Rubric

**Repo:** <https://github.com/fixmeseb/Theta-IDE>

Scale: **0** none · **1** a little · **2** somewhat · **3** a lot.
Every row carries a link as evidence. The tutor spot-checks five rows at the demo,
so each score below is one a reviewer can verify from the repository.

Figures measured on commit `2a9802d`: 752 commits, 1365 tests, 70% line coverage.

---

## Teamwork, visible in GitHub

| Criterion | Score | Evidence |
|---|---|---|
| Many commits (exemplar has 467) | **3** | [752 commits on main](https://github.com/fixmeseb/Theta-IDE/commits/main) — 61% above the exemplar |
| Commits come from different people | **3** | [Contributors](https://github.com/fixmeseb/Theta-IDE/graphs/contributors) — 6 identities, 5 people |
| Workload is spread over the whole team | **1** | [Contributors](https://github.com/fixmeseb/Theta-IDE/graphs/contributors). 561 / 59 / 58 / 41 / 31 / 2. The lead has 75% of commits, roughly 8x the next contributor — beyond the "3x more productive" the rubric allows for |
| Team members work across many parts, not one silo each | **?** | *See note below — needs your judgement* |
| Short release cycles: frequent small commits, merged promptly | **3** | [25 pull requests, all merged or closed, none open](https://github.com/fixmeseb/Theta-IDE/pulls?q=is%3Apr) |
| Whole team uses the same tools: config files in repo, updated by many | **3** | [`pyproject.toml`](https://github.com/fixmeseb/Theta-IDE/blob/main/pyproject.toml) (`[tool.ruff]`, `[tool.mypy]`, `[tool.coverage]`), [`.pre-commit-config.yaml`](https://github.com/fixmeseb/Theta-IDE/blob/main/.pre-commit-config.yaml), [`ci.yml`](https://github.com/fixmeseb/Theta-IDE/blob/main/.github/workflows/ci.yml) |
| Whole team can run the system on their own machine | **?** | *Demo-time check — only the team can answer* |

## Issues and discussion

| Criterion | Score | Evidence |
|---|---|---|
| Issue reports: there are many | **1** | [3 issues total](https://github.com/fixmeseb/Theta-IDE/issues?q=is%3Aissue) |
| Issues are being closed | **3** | All 3 closed, none open |
| Issues are discussed before they are closed | **0** | None of the 3 carries a single comment. Design talk happened in Discord and was never summarised back |
| A large share of issues relate to handling failing cases | **3** | All 3 are failing-test reports: [#11](https://github.com/fixmeseb/Theta-IDE/issues/11), [#12](https://github.com/fixmeseb/Theta-IDE/issues/12), [#13](https://github.com/fixmeseb/Theta-IDE/issues/13) — each opened on a failure and closed by a fix |
| Chat channel exists and is active; poster QR points at it | **?** | *Discord is active; score depends on whether the QR is on the poster* |

## Tests

| Criterion | Score | Evidence |
|---|---|---|
| Tests exist: 50 is good, 100 is better | **3** | [1365 tests](https://github.com/fixmeseb/Theta-IDE/tree/main/tests) across 107 files, expected and failure cases both |
| Tests are a serious share of the code base (~30%) | **3** | 24,817 of 85,758 Python lines are tests — **28%** |
| Tests run automatically on every commit | **3** | [CI badge](https://github.com/fixmeseb/Theta-IDE/actions/workflows/ci.yml) — lint, test and typecheck on push and PR |
| Coverage is measured; the badge shows the number | **2** | **70%** measured. Badge shows the number but is a constant, not a live Codecov feed — see Known Gaps |

## Docs

| Criterion | Score | Evidence |
|---|---|---|
| Docs are generated, and not ugly | **1** | No generator (no Sphinx, MkDocs or pdoc). Docs are hand-written Markdown plus an in-app browser |
| Docs, the WHAT: per class/function descriptions | **2** | 1091 of 2242 public classes and functions carry docstrings — **48%** |
| Docs, the HOW: mini-tutorials with worked examples | **3** | [21 in-app articles](https://github.com/fixmeseb/Theta-IDE/blob/main/frontend/plugins/core/documentation/content.py) (1490 lines), plus [`COMPONENT_PLUGIN_GUIDE.md`](https://github.com/fixmeseb/Theta-IDE/blob/main/docs/COMPONENT_PLUGIN_GUIDE.md) and [`WORKFLOW_GUIDE.md`](https://github.com/fixmeseb/Theta-IDE/blob/main/docs/WORKFLOW_GUIDE.md) |
| Docs, the WHY: the docs tell a story | **?** | *Judgement call — "Welcome & Architecture Overview" and "System Architecture" attempt it* |
| Teaser video (12–30 s) at the TOP of README | **3** | [Embedded directly under the badges](https://github.com/fixmeseb/Theta-IDE#readme) |
| Feature walk-through video (2–5 min), linked in README | **0** | Only the teaser is present |

## Standard files and bling

| Criterion | Score | Evidence |
|---|---|---|
| README.md: all badges above, all working | **2** | [README](https://github.com/fixmeseb/Theta-IDE#readme) — 8 badges, every URL verified to return 200. Short of 3 only because coverage is not a live feed and there is no DOI |
| .gitignore | **3** | [`.gitignore`](https://github.com/fixmeseb/Theta-IDE/blob/main/.gitignore) |
| INSTALL.md | **3** | [`INSTALL.md`](https://github.com/fixmeseb/Theta-IDE/blob/main/INSTALL.md) |
| LICENSE.md | **3** | [`LICENSE.md`](https://github.com/fixmeseb/Theta-IDE/blob/main/LICENSE.md) — MIT |
| CODE-OF-CONDUCT.md | **3** | [`CODE-OF-CONDUCT.md`](https://github.com/fixmeseb/Theta-IDE/blob/main/CODE-OF-CONDUCT.md) |
| CONTRIBUTING.md | **3** | [`CONTRIBUTING.md`](https://github.com/fixmeseb/Theta-IDE/blob/main/CONTRIBUTING.md) |
| Style checker, formatter, syntax checker: configs + badges | **3** | Ruff and mypy configured in [`pyproject.toml`](https://github.com/fixmeseb/Theta-IDE/blob/main/pyproject.toml), enforced by [CI](https://github.com/fixmeseb/Theta-IDE/blob/main/.github/workflows/ci.yml), badged in the README |
| DOI badge from Zenodo | **0** | Not yet minted — see Known Gaps |
| THIRD_PARTY_LIBRARIES.md: every dependency with its licence | **3** | [`THIRD_PARTY_LIBRARIES.md`](https://github.com/fixmeseb/Theta-IDE/blob/main/THIRD_PARTY_LIBRARIES.md) — 38 dependencies, generated from installed metadata, and it surfaces that PyQt6 is GPL-3.0 in an MIT project |
| AI use is visible and disclosed | **2** | [`.agents/AGENTS.md`](https://github.com/fixmeseb/Theta-IDE/blob/main/.agents) in the repo; human checking is evidenced by review threads on [the pull requests](https://github.com/fixmeseb/Theta-IDE/pulls?q=is%3Apr). No `.coderabbit.yaml` or equivalent bot config |
| Repo holds no keys or passwords | **3** | Scanned for hardcoded credentials; none found. CI uses `secrets.CODECOV_TOKEN` |

---

## Subtotal

33 rows in the rubric. **29 scored here, 4 left for the team** (marked **?**),
because they depend on a demo, the poster, or a judgement only the team can make.

**Scored sum: 68 / 87** (76%).

Filling the four open rows at their likely values would put the total near
78 / 99. The gaps below are worth 14 points between them, and most are hours
of work rather than days.

## Known gaps, and what closes them

| Gap | Cost | Closes |
|---|---|---|
| No Zenodo DOI | Link Zenodo to the repo, cut a release | 1 row, 0 → 3 |
| Coverage badge is a constant | Link Codecov to `fixmeseb/Theta-IDE`, set `CODECOV_TOKEN` | lifts 2 rows to 3 |
| 3 issues, none discussed | Open issues for known defects and discuss before closing | 2 rows, worth 5 points |
| No feature walk-through video | Record 2–5 min, link in README | 1 row, 0 → 3 |
| No generated docs | Add pdoc or MkDocs over the existing docstrings | 1 row, 1 → 3 |

The issue rows are the cheapest real points: four defects are already identified
and unreported — a modal dialog that hangs `test_export_dialog.py`, an abort in
`test_ui_adjustments.py`, the Hub failing silently when Python has no CA bundle,
and `standard-figures` not opting into `draws_metric_series`. Filing and
discussing those is honest work that moves two rows.

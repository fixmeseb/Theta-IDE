"""Results panel (frontend/results): explorer, one chart per metric, hiding plots, axes, smoothing."""

import os
import sys
import tempfile
from pathlib import Path

import pytest

pytest.importorskip("PyQt6")

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PyQt6.QtCore import Qt  # noqa: E402
from PyQt6.QtWidgets import QApplication, QTabWidget  # noqa: E402

QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)
app = QApplication.instance() or QApplication(sys.argv[:1])

from frontend.results.data import Source  # noqa: E402
from frontend.results.panel import ResultsPanel  # noqa: E402

PPO_ROWS = [  # Lightning PPO: eval and loss values on separate rows
    {"epoch": "", "eval/reward": "20", "losses/total_loss": "", "time/total": "1", "transitions": "1000"},
    {"epoch": "0", "eval/reward": "", "losses/total_loss": "0.5", "time/total": "", "transitions": "1000"},
    {"epoch": "", "eval/reward": "60", "losses/total_loss": "", "time/total": "2", "transitions": "2000"},
    {"epoch": "1", "eval/reward": "", "losses/total_loss": "0.25", "time/total": "", "transitions": "2000"},
]
SB3_ROWS = [  # SB3 through ContractLogger: its own loss names
    {"step": "1", "transitions": "1000", "eval/reward": "15", "losses/train/value_loss": "3.5"},
    {"step": "2", "transitions": "2000", "eval/reward": "35", "losses/train/value_loss": "2.0"},
]
RUNS = [
    {"group": "cartpole", "experiment_id": "ppo_run", "agents": [{"name": "ppo"}]},
    {"group": "sb3", "experiment_id": "sb3_run", "agents": [{"name": "ppo"}]},
    {"group": "sb3", "experiment_id": "never_logged", "agents": []},  # nothing to plot: not listed
]
OLD_PPO_ROWS = [{"eval/reward": "5", "transitions": "1000"}]


def _png() -> bytes:
    from PyQt6.QtCore import QBuffer, QIODevice
    from PyQt6.QtGui import QColor, QImage

    image = QImage(8, 6, QImage.Format.Format_RGB32)
    image.fill(QColor("#83a598"))
    buffer = QBuffer()
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    image.save(buffer, "PNG")
    return bytes(buffer.data())


# Every file each run produced, as results/-relative path -> (kind, bytes); the manifest lists these.
FILES = {
    "cartpole/ppo_run": {
        "logs/cartpole/ppo_run/config.yaml": ("config", b"seed: 1\nenv:\n  name: cartpole\nlr: 0.001\n"),
        "logs/cartpole/ppo_run/ppo/version_1/metrics.csv": ("metrics", b"transitions,eval/reward\n1000,20\n2000,60\n"),
        "plots/cartpole/ppo_run/time_report.csv": ("table", b"Method,Avg_Time_Seconds,Runs\nppo,14.8,1\niql,9.5,2\n"),
        "plots/cartpole/ppo_run/losses/ppo/losses_total_loss.png": ("figure", None),  # filled with _png() below
        "plots/cartpole/ppo_run/hyperparameters_report.md": ("report", b"# Hyperparameters\n\n| lr | 0.001 |\n"),
        "checkpoints/cartpole/ppo_run/ppo/best_model.ckpt": ("checkpoint", b"\x00weights"),
    },
    "sb3/sb3_run": {
        "logs/sb3/sb3_run/config.yaml": ("config", b"seed: 2\nenv:\n  name: cartpole\nlr: 0.001\n"),
    },
}


def _manifest(run_key, versions):
    group, exp = run_key.split("/")
    artifacts = [
        {"path": path, "kind": kind, "size": len(data or b"x"), "modified": 1.0, "agent": None, "version": None}
        for path, (kind, data) in FILES.get(run_key, {}).items()
    ]
    return {"group": group, "experiment_id": exp, "agents": {"ppo": versions}, "artifacts": artifacts}


class FakeBackend:
    """Answers like the real API, synchronously. `offline` makes every request fail."""

    def __init__(self):
        self.requests = []
        self.offline = False

    def get(self, path, callback):
        self.requests.append(path)
        if self.offline:
            callback(None, "Connection refused")
        elif path == "/api/runs":
            callback({"runs": RUNS}, None)
        elif path == "/api/runs/cartpole/ppo_run/manifest":
            callback(_manifest("cartpole/ppo_run", ["version_0", "version_1"]), None)
        elif path.endswith("/manifest"):
            group, exp = path.split("/")[3:5]
            callback(_manifest(f"{group}/{exp}", ["version_0"]), None)
        elif path == "/api/runs/cartpole/ppo_run/ppo/metrics?version=version_0":
            callback({"metrics": OLD_PPO_ROWS, "columns": ["eval/reward", "transitions"]}, None)
        elif path.startswith("/api/runs/cartpole/ppo_run/ppo/metrics"):
            callback({"metrics": PPO_ROWS, "columns": list(PPO_ROWS[0])}, None)
        elif path.startswith("/api/runs/sb3/sb3_run/ppo/metrics"):
            callback({"metrics": SB3_ROWS, "columns": list(SB3_ROWS[0])}, None)
        else:
            callback(None, f"HTTP 404: {path}")

    def get_bytes(self, path, callback):
        self.requests.append(path)
        group, exp, rest = path.removeprefix("/api/runs/").split("/", 2)
        artifact = rest.removeprefix("files/")
        kind, data = FILES.get(f"{group}/{exp}", {}).get(artifact, (None, None))
        if kind == "figure":
            data = _png()
        callback(data, None) if data is not None else callback(None, f"HTTP 404: {path}")


class FakeSettings:
    def __init__(self, data=None):
        self.data = data or {}

    def get(self, *keys, default=None):
        node = self.data
        for key in keys:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    def set(self, *keys_and_value):
        *keys, value = keys_and_value
        node = self.data
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        node[keys[-1]] = value


def tree_items(panel):
    tree = panel.explorer.tree
    stack = [tree.topLevelItem(i) for i in range(tree.topLevelItemCount())]
    items = {}
    while stack:
        item = stack.pop()
        items[item.text(0)] = item
        stack.extend(item.child(i) for i in range(item.childCount()))
    return items


def make_panel(settings=None):
    backend = FakeBackend()
    panel = ResultsPanel(backend, settings)
    panel.refresh()
    return panel, backend


def tick(panel, *names):
    items = tree_items(panel)
    for name in names:
        items[name].setCheckState(0, Qt.CheckState.Checked)


def series_names(panel, metric):
    return [title for title, _, _ in panel.grid.cards[metric].chart.series]


def test_explorer_lists_runs_that_logged_metrics():
    panel, _ = make_panel()
    items = tree_items(panel)
    assert {"cartpole", "sb3", "ppo_run", "sb3_run"} <= set(items)
    assert "never_logged" not in items
    assert panel.status.text().startswith("2 runs in results/logs")
    assert panel.grid.cards == {}  # nothing ticked yet


def test_ticking_runs_plots_every_metric_they_logged():
    panel, _ = make_panel()
    tick(panel, "ppo_run", "sb3_run")  # ticking an experiment ticks its agents
    # One chart per metric across both engines, nothing whitelisted; reward is shared by both runs.
    assert set(panel.grid.cards) == {"eval/reward", "losses/total_loss", "losses/train/value_loss", "time/total"}
    assert series_names(panel, "eval/reward") == ["ppo_run / ppo (version_1)", "sb3_run / ppo"]
    assert series_names(panel, "losses/train/value_loss") == ["sb3_run / ppo"]
    reward = dict((t, p) for t, p, _ in panel.grid.cards["eval/reward"].chart.series)
    assert reward["sb3_run / ppo"] == [{"step": 1000.0, "eval/reward": 15.0}, {"step": 2000.0, "eval/reward": 35.0}]
    assert "sb3_run / ppo" in panel.legend.text()


def test_multi_version_agents_offer_each_version_with_the_newest_ticked():
    panel, _ = make_panel()
    tick(panel, "ppo_run")
    items = tree_items(panel)
    assert items["version_1"].checkState(0) == Qt.CheckState.Checked
    assert items["version_0"].checkState(0) == Qt.CheckState.Unchecked
    assert panel.sources == [Source("cartpole", "ppo_run", "ppo", "version_1")]
    items["version_0"].setCheckState(0, Qt.CheckState.Checked)  # overlay the older version too
    assert series_names(panel, "eval/reward") == ["ppo_run / ppo (version_0)", "ppo_run / ppo (version_1)"]


def test_hiding_plots_is_remembered_in_settings():
    settings = FakeSettings()
    panel, _ = make_panel(settings)
    tick(panel, "sb3_run")
    panel.grid.cards["losses/train/value_loss"].hide_button.click()
    assert "losses/train/value_loss" not in panel.grid.cards
    assert settings.data == {"results": {"hidden_plots": ["losses/train/value_loss"]}}
    assert "1 hidden" in panel.plots_button.text()
    # A new panel (e.g. after restarting the app) keeps it hidden; Show all brings it back.
    again, _ = make_panel(settings)
    tick(again, "sb3_run")
    assert set(again.grid.cards) == {"eval/reward"}
    again.show_all()
    assert set(again.grid.cards) == {"eval/reward", "losses/train/value_loss"}
    assert settings.data["results"]["hidden_plots"] == []


def test_plots_menu_lists_every_metric_with_its_visibility():
    panel, _ = make_panel(FakeSettings({"results": {"hidden_plots": ["eval/reward"]}}))
    tick(panel, "sb3_run")
    panel._fill_plots_menu()
    actions = {a.text(): a for a in panel.plots_menu.actions() if a.isCheckable()}
    assert actions["Episode reward  ·  eval/reward"].isChecked() is False
    assert actions["Value loss (SB3)  ·  losses/train/value_loss"].isChecked() is True
    actions["Episode reward  ·  eval/reward"].setChecked(True)  # show it again from the menu
    assert "eval/reward" in panel.grid.cards


def test_x_axis_choice_and_smoothing():
    panel, _ = make_panel()
    tick(panel, "ppo_run")
    axes = [panel.axis_select.itemData(i) for i in range(panel.axis_select.count())]
    assert axes == ["transitions", "epoch", "time/total"]
    panel.axis_select.setCurrentIndex(axes.index("epoch"))
    ((_, loss, _),) = panel.grid.cards["losses/total_loss"].chart.series
    assert [p["step"] for p in loss] == [0.0, 1.0]
    panel.smoothing_slider.setValue(50)
    ((_, loss, _),) = panel.grid.cards["losses/total_loss"].chart.series
    assert [p["losses/total_loss"] for p in loss] == [0.5, 0.375]


def test_offline_backend_explains_itself():
    backend = FakeBackend()
    backend.offline = True
    panel = ResultsPanel(backend)
    panel.refresh()
    assert "Could not load" in panel.status.text()
    panel.set_offline()
    assert "Backend offline" in panel.status.text()


def test_window_hosts_the_panel_beside_the_run_list():
    from frontend.app import Window

    with tempfile.TemporaryDirectory() as data_dir:
        window = Window(Path(data_dir))
        try:
            assert isinstance(window.results_panel, QTabWidget)
            assert window.results_panel.widget(0) is window.results_view
            assert window.results_panel.tabText(1) == "Run list"
            assert window.table is not None  # the run table and its actions are still there
        finally:
            window.close()


# ── Phase 4: every data point ────────────────────────────────────────────────


def browser_select(panel, tab, suffix):
    browser = panel.browser(tab)
    run_key, artifact = next((k, a) for k, a in browser.artifacts() if a["path"].endswith(suffix))
    assert browser.select(run_key, artifact["path"])
    return browser


def test_point_inspector_shows_every_metric_of_every_run():
    panel, _ = make_panel()
    tick(panel, "ppo_run", "sb3_run")
    panel.grid.cards["eval/reward"].chart.point_hovered.emit(2000.0)
    values = {s.experiment_id: v for s, v in panel.inspector.values.items()}
    # Lightning wrote reward and loss on separate rows; the inspector merges them for the step.
    assert values["ppo_run"] == {
        "transitions": 2000.0,
        "eval/reward": 60.0,
        "time/total": 2.0,
        "epoch": 1.0,
        "losses/total_loss": 0.25,
    }
    assert values["sb3_run"]["losses/train/value_loss"] == 2.0
    assert panel.inspector.table.rowCount() == 7  # every metric of both runs, including the axes
    # Between logged points, each run shows its nearest one and the caption says so.
    panel.grid.point_hovered.emit(1400.0)
    assert {s.experiment_id: v["transitions"] for s, v in panel.inspector.values.items()} == {
        "ppo_run": 1000.0,
        "sb3_run": 1000.0,
    }
    assert "nearest logged point" in panel.inspector.caption.text()


def test_tables_tab_shows_every_row_and_plots_numeric_columns():
    panel, _ = make_panel()
    tick(panel, "ppo_run")
    names = [a["path"].rsplit("/", 1)[1] for _, a in panel.browser("Tables").artifacts()]
    assert names == ["metrics.csv", "time_report.csv"]  # raw metrics.csv is a table too
    browser = browser_select(panel, "Tables", "time_report.csv")
    viewer = browser.viewers["table"]
    assert browser.stack.currentWidget() is viewer
    assert (viewer.table.rowCount(), viewer.table.columnCount()) == (2, 3)
    assert viewer.table.item(1, 0).text() == "iql"
    assert viewer.plot.isVisibleTo(browser) and list(viewer.columns) == ["Avg_Time_Seconds", "Runs"]
    assert "2 rows × 3 columns" in browser.detail.text()


def test_figures_reports_and_text_are_previewed():
    panel, _ = make_panel()
    tick(panel, "ppo_run")
    figures = browser_select(panel, "Figures", "losses_total_loss.png")
    assert figures.stack.currentWidget() is figures.viewers["figure"] and "8 × 6 px" in figures.detail.text()
    assert figures.title.text() == "plots/losses/ppo/losses_total_loss.png"  # path inside the run's folders
    reports = browser_select(panel, "Reports", "hyperparameters_report.md")
    assert "Hyperparameters" in reports.viewers["text"].browser.toPlainText()


def test_config_tab_lists_only_settings_that_differ():
    panel, _ = make_panel()
    tick(panel, "ppo_run")
    panel.browser("Config")  # the Config tab is built when first opened
    assert len(panel.config_diff.rows) == 3  # one run: every setting
    tick(panel, "sb3_run")
    assert panel.config_diff.rows == [("seed", [1, 2])]  # env.name and lr match, so they are hidden
    assert "1 settings differ across 2 runs" in panel.config_diff.caption.text()


def test_files_tab_reaches_every_artifact_and_saves_copies(tmp_path):
    panel, _ = make_panel()
    tick(panel, "ppo_run", "sb3_run")
    listed = {a["path"] for _, a in panel.browser("Files").artifacts()}
    assert listed == {path for files in FILES.values() for path in files}  # every file of both runs
    browser = browser_select(panel, "Files", "best_model.ckpt")
    assert browser.stack.currentWidget() is browser.info
    assert "No preview for this kind of file" in browser.info.text.text()
    target = tmp_path / "copy.ckpt"
    browser.save_copy(str(target))
    assert target.read_bytes() == b"\x00weights"


def test_artifact_tabs_are_built_when_first_opened():
    panel, _ = make_panel()
    tick(panel, "ppo_run")
    assert panel.browsers == {} and panel.config_diff is None  # nothing built up front
    panel.tabs.setCurrentIndex([panel.tabs.tabText(i) for i in range(panel.tabs.count())].index("Figures"))
    assert list(panel.browsers) == ["Figures"]
    # A tab opened after runs were ticked lists their files straight away.
    assert [a["path"].rsplit("/", 1)[1] for _, a in panel.browsers["Figures"].artifacts()] == ["losses_total_loss.png"]


def test_files_too_large_to_preview_are_not_downloaded():
    panel, backend = make_panel()
    tick(panel, "ppo_run")
    browser = panel.browser("Files")
    run_key, artifact = next((k, a) for k, a in browser.artifacts() if a["path"].endswith("time_report.csv"))
    artifact["size"] = 10**9
    before = len(backend.requests)
    browser.select(run_key, artifact["path"])
    assert "Too large to preview" in browser.info.text.text() and len(backend.requests) == before


# ── Phase 5: comparisons ─────────────────────────────────────────────────────


def set_combine(panel, mode):
    panel.combine_select.setCurrentIndex(panel.combine_select.findData(mode))


def test_combining_versions_draws_one_mean_curve_with_a_band():
    panel, _ = make_panel()
    tick(panel, "ppo_run")
    tree_items(panel)["version_0"].setCheckState(0, Qt.CheckState.Checked)  # both versions ticked
    assert series_names(panel, "eval/reward") == ["ppo_run / ppo (version_0)", "ppo_run / ppo (version_1)"]
    set_combine(panel, "std")
    chart = panel.grid.cards["eval/reward"].chart
    ((name, points, _),) = chart.series
    assert name == "ppo_run / ppo (mean of 2)" and chart.band == "__spread"
    first, second = points
    # version_0 reward 5 and version_1 reward 20 at step 1000: mean 12.5, sample std 10.61.
    assert (first["step"], first["eval/reward"], first["n"]) == (1000.0, 12.5, 2)
    assert first["__spread"] == pytest.approx(10.6066, abs=1e-3)
    assert (second["eval/reward"], second["n"], second["__spread"]) == (60.0, 1, 0.0)  # only version_1 here
    set_combine(panel, "sem")
    ((_, points, _),) = chart.series
    assert points[0]["__spread"] == pytest.approx(7.5)  # std / sqrt(2)
    set_combine(panel, None)
    assert chart.band is None and len(chart.series) == 2


def test_summary_tab_lists_every_series_and_metric():
    panel, _ = make_panel()
    tick(panel, "sb3_run")
    rows = {row[1]: row for row in panel.summary.rows}
    reward = rows["eval/reward"]
    # Series, Metric, Final, Best, Best at, Min, Max, Mean, Area under curve, Points
    assert reward == ["sb3_run / ppo", "eval/reward", 35.0, 35.0, 2000.0, 15.0, 35.0, 25.0, 25000.0, 2]
    loss = rows["losses/train/value_loss"]
    assert (loss[3], loss[4]) == (2.0, 2000.0)  # a loss: lower is better
    assert panel.summary.table.rowCount() == 2


def test_summary_threshold_reports_when_each_series_got_there():
    panel, _ = make_panel()
    tick(panel, "ppo_run", "sb3_run")
    summary = panel.summary
    assert not summary.threshold.isEnabled()  # thresholds need one metric
    summary.metric_select.setCurrentIndex(summary.metric_select.findData("eval/reward"))
    summary.threshold.setText("30")
    assert summary.headers[-1] == "Reaches threshold at"
    reached = {row[0]: row[-1] for row in summary.rows}
    assert reached == {"ppo_run / ppo (version_1)": 2000.0, "sb3_run / ppo": 2000.0}
    summary.threshold.setText("100")
    assert all(row[-1] is None for row in summary.rows)  # nobody got there


def test_exports_write_curves_summary_and_charts(tmp_path):
    panel, _ = make_panel()
    tick(panel, "sb3_run")
    panel.set_hidden("losses/train/value_loss", True)  # hidden charts are not exported
    assert panel.export_to("curves", str(tmp_path / "curves.csv"))
    assert (tmp_path / "curves.csv").read_text(encoding="utf-8").splitlines() == [
        "series,metric,transitions,value",
        "sb3_run / ppo,eval/reward,1000.0,15.0",
        "sb3_run / ppo,eval/reward,2000.0,35.0",
    ]
    assert panel.export_to("summary", str(tmp_path / "summary.csv"))
    summary = (tmp_path / "summary.csv").read_text(encoding="utf-8").splitlines()
    assert summary[0].startswith("Series,Metric,Final,Best") and len(summary) == 3
    panel.resize(900, 700)
    assert panel.export_to("png", str(tmp_path / "charts.png"))
    assert (tmp_path / "charts.png").read_bytes()[:4] == bytes([0x89]) + b"PNG"


def test_combined_curves_export_their_spread(tmp_path):
    panel, _ = make_panel()
    tick(panel, "ppo_run")
    tree_items(panel)["version_0"].setCheckState(0, Qt.CheckState.Checked)
    set_combine(panel, "sem")
    lines = panel.curves_csv().splitlines()
    assert lines[0] == "series,metric,transitions,value,sem,runs"
    assert lines[1] == "ppo_run / ppo (mean of 2),eval/reward,1000.0,12.5,7.5,2"


def test_load_in_builder_needs_exactly_one_ticked_run():
    calls = []
    backend = FakeBackend()
    panel = ResultsPanel(backend, load_config=lambda run_key: calls.append(run_key) or None)
    panel.refresh()
    panel.browser("Config")
    tick(panel, "ppo_run")
    assert panel.load_button.isEnabled()
    panel.load_in_builder()
    assert calls == ["cartpole/ppo_run"] and "Loaded the saved config" in panel.status.text()
    tick(panel, "sb3_run")
    assert not panel.load_button.isEnabled()
    panel.load_in_builder()
    assert calls == ["cartpole/ppo_run"] and "exactly one run" in panel.status.text()


def test_window_loads_results_runs_into_the_builder_only_when_it_has_their_config():
    from frontend.app import Window

    with tempfile.TemporaryDirectory() as data_dir:
        window = Window(Path(data_dir))
        try:
            assert "No builder config" in window.load_results_config("cartpole/unknown_run")
            run = window.runs[0]  # an example run, given a backend identity as if the app had launched it
            run.update(simulated=False, backend={"group": "thetaide", "experiment_id": "exp_1", "job_id": "j"})
            assert window.load_results_config("thetaide/exp_1") is None
            assert window.tabs.currentWidget() is window.config_panel
        finally:
            window.close()

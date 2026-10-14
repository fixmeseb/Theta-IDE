"""Training monitor live views: status strip, every chart, system use, logs, feed and files."""

import os
import sys
import tempfile
from pathlib import Path

import pytest

pytest.importorskip("PyQt6")

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PyQt6.QtCore import Qt  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)
app = QApplication.instance() or QApplication(sys.argv[:1])

from frontend.app import Window  # noqa: E402
from frontend.live_run import LiveRun, LiveRunCache, format_duration  # noqa: E402
from frontend.model import disk_run  # noqa: E402
from frontend.monitor.live_view import (  # noqa: E402
    ChartsView,
    FeedView,
    FilesView,
    LogView,
    StatusStrip,
    SystemView,
)
from src.app import telemetry  # noqa: E402
from src.app.telemetry.records import record  # noqa: E402

T0 = 1_760_000_000.0


def timed(rec, t):
    rec["t"] = t
    return rec


def sample_live(budget=10000):
    """A run 40 s in: four evaluations, losses, system samples, a checkpoint and a warning."""
    live = LiveRun(budget=budget)
    records = [timed(record("start", run={"version": "version_0", "pid": 42, "started": T0}, config={}), T0)]
    for i, reward in enumerate([10.0, 30.0, 25.0, 60.0]):
        data = {"eval/reward": reward, "losses/total_loss": 2.0 / (i + 1), "transitions": 1000.0 * (i + 1)}
        records.append(timed(record("metrics", step=i, data=data), T0 + 10 * (i + 1)))
        records.append(timed(record("system", data={"cpu_percent": 100.0 + i, "rss_mb": 500.0 + i}), T0 + 10 * i + 5))
    records.append(
        timed(record("artifact", path="ckpt/best_model.ckpt", kind="checkpoint", data={"best": True}), T0 + 41)
    )
    records.append(
        timed(record("artifact", path="ckpt/best_model.ckpt", kind="checkpoint", data={"best": True}), T0 + 42)
    )
    records.append(timed(record("event", kind="non_finite", level="warning", message="losses/x is nan"), T0 + 43))
    live.add(records, now=T0 + 43)
    return live


# ── The views on their own ───────────────────────────────────────────────────


def test_status_strip_shows_speed_time_left_and_a_stall():
    strip = StatusStrip()
    lives = {"ppo": sample_live()}
    strip.show_run("running", lives, None, live=True, now=T0 + 50)
    assert strip.badge.text() == "RUNNING"
    assert strip.fields["elapsed"].text() == "50s"
    assert strip.fields["speed"].text() == "100 steps/s"  # 3,000 transitions over the last 30 s
    assert strip.fields["eta"].text() == format_duration(60)  # 6,000 left at 100/s
    assert strip.fields["updated"].text() == "7s ago"
    strip.show_run("running", lives, None, live=True, now=T0 + 43 + 600)
    assert strip.badge.text() == "STALLED" and "no new data" in strip.fields["updated"].text()
    strip.show_run("completed", lives, None, live=False, now=T0 + 600)
    assert strip.badge.text() == "COMPLETED" and strip.fields["speed"].text() == "—"


def test_charts_view_plots_every_metric_per_agent_and_hides_on_request():
    view = ChartsView()
    second = sample_live()
    view.show_lives({"ppo": sample_live(), "dqn": second})
    assert set(view.grid.cards) == {"eval/reward", "losses/total_loss"}  # counters are axes, not charts
    chart = view.grid.cards["eval/reward"].chart
    assert [name for name, _, _ in chart.series] == ["ppo", "dqn"]
    assert [p["step"] for p in chart.series[0][1]] == [1000.0, 2000.0, 3000.0, 4000.0]
    assert "x-axis: environment steps" in view.caption.text()
    view.grid.cards["losses/total_loss"].hide_button.click()
    assert set(view.grid.cards) == {"eval/reward"} and view.show_all.isVisibleTo(view)
    view.show_all.click()
    assert set(view.grid.cards) == {"eval/reward", "losses/total_loss"}


def test_system_view_plots_samples_over_time_without_hide_buttons():
    view = SystemView()
    view.show_lives({"ppo": sample_live()})
    assert list(view.grid.cards) == ["system/cpu_percent", "system/rss_mb"]
    card = view.grid.cards["system/cpu_percent"]
    assert card.chart.title.startswith("CPU") and card.hide_button.isHidden()
    assert [p["step"] for p in card.chart.series[0][1]] == [0.0, 10.0, 20.0, 30.0]
    view.show_lives({})
    assert view.grid.cards == {} and "No resource samples" in view.grid.empty.text()


def test_feed_lists_events_newest_first_with_start_and_finish():
    live = sample_live()
    live.add([timed(record("finish", status="failed", summary={}), T0 + 60)], now=T0 + 60)
    entries = FeedView.entries({"ppo": live})
    assert entries[0][2:] == ("error", "Run finished: failed")
    assert entries[-1][3] == "Run started (version_0, pid 42)"
    assert [e[3] for e in entries if e[2] == "warning"] == ["losses/x is nan"]
    assert any(e[3].startswith("New best eval/reward 60") for e in entries)
    view = FeedView()
    view.show_lives({"ppo": live})
    assert view.tree.topLevelItemCount() == len(entries)


def test_files_view_lists_each_file_once():
    view = FilesView()
    view.show_lives({"ppo": sample_live()}, None)
    assert view.tree.topLevelItemCount() == 1
    item = view.tree.topLevelItem(0)
    assert (item.text(0), item.text(1)) == ("ckpt/best_model.ckpt", "checkpoint  (best)")
    assert not view.open_folder.isEnabled()


def test_log_view_filters_problems_and_keeps_new_lines():
    view = LogView()
    view.set_lines(["start", "WARNING: slow", "Traceback (most recent call last):", "done"])
    assert view.text.toPlainText().splitlines()[0] == "start"
    view.problems_only.setChecked(True)
    assert view.text.toPlainText().splitlines() == ["WARNING: slow", "Traceback (most recent call last):"]
    view.append(["fine", "RuntimeError: boom"])
    assert view.text.toPlainText().splitlines()[-1] == "RuntimeError: boom"
    view.problems_only.setChecked(False)
    assert view.text.toPlainText().splitlines()[-2:] == ["fine", "RuntimeError: boom"]
    view.set_lines([], "nothing captured")
    assert view.empty.text() == "nothing captured"


# ── In the window ────────────────────────────────────────────────────────────


@pytest.fixture
def window():
    with tempfile.TemporaryDirectory() as data_dir:
        win = Window(Path(data_dir))
        win.jobs.logs_dir = Path(data_dir) / "logs"
        try:
            yield win
        finally:
            win.close()


def open_view(window, title):
    views = window.monitor.views
    views.setCurrentIndex([views.tabText(i) for i in range(views.count())].index(title))
    window.monitor.refresh_live()


def test_a_followed_job_fills_the_live_views_as_records_arrive(window):
    run = disk_run({"group": "g", "experiment_id": "e", "agents": [{"name": "ppo"}], "config": {}, "metadata": {}})
    run["status"] = "running"
    run["backend"]["job_id"] = "job1"
    window.runs.insert(0, run)
    window.select_run(run)
    live = sample_live()
    window.live_runs[("job1", "ppo")] = live
    open_view(window, "Charts")
    assert set(window.monitor.charts_view.grid.cards) == {"eval/reward", "losses/total_loss"}
    window.on_job_records("job1", "ppo", [record("metrics", step=9, data={"val/f1": 0.5})], False)
    window.monitor.refresh_live()
    assert "val/f1" in window.monitor.charts_view.grid.cards
    open_view(window, "Feed")
    assert window.monitor.feed_view.tree.topLevelItemCount() >= 3


def test_a_finished_run_is_read_from_disk_when_a_live_view_opens(window):
    version_dir = window.jobs.logs_dir / "g" / "old" / "ppo" / "version_0"
    run_log = telemetry.Run(version_dir, ["csv", "stream"], system_interval=0)
    for i in range(3):
        run_log.log({"eval/reward": 10.0 * i}, step=i, transitions=100 * i)
    run_log.system({"cpu_percent": 50.0, "rss_mb": 200.0})
    run_log.finish()
    run = disk_run({"group": "g", "experiment_id": "old", "agents": [{"name": "ppo"}], "config": {}, "metadata": {}})
    window.runs.insert(0, run)
    window.select_run(run)
    assert window.live_for_run(run, load=False) == {}  # the Overview alone reads nothing
    open_view(window, "System")
    assert list(window.monitor.system_view.grid.cards) == ["system/cpu_percent", "system/rss_mb"]
    open_view(window, "Charts")
    assert list(window.monitor.charts_view.grid.cards) == ["eval/reward"]
    assert window.monitor.status_strip.badge.text() == run["status"].upper()
    open_view(window, "Logs")
    assert "started outside ThetaIDE" in window.monitor.log_view.empty.text()
    open_view(window, "Files")
    assert window.monitor.files_view.open_folder.isEnabled()


def test_the_disk_cache_rereads_only_changed_runs(tmp_path):
    version_dir = tmp_path / "g" / "e" / "ppo" / "version_0"
    run_log = telemetry.Run(version_dir, ["stream"], system_interval=0)
    run_log.log({"eval/reward": 1.0}, step=1)
    cache = LiveRunCache()
    first = cache.load(tmp_path, "g", "e", ["ppo"])["ppo"]
    assert cache.load(tmp_path, "g", "e", ["ppo"])["ppo"] is first
    run_log.log({"eval/reward": 2.0}, step=2)
    run_log.finish()
    again = cache.load(tmp_path, "g", "e", ["ppo"])["ppo"]
    assert again is not first and again.latest("eval/reward") == 2.0 and again.is_finished()
    assert cache.cached(tmp_path, "g", "e", ["ppo", "missing"]) == {"ppo": again}


def test_output_missed_while_another_run_was_shown_is_read_again(window):
    first = disk_run({"group": "g", "experiment_id": "a", "agents": [{"name": "ppo"}], "config": {}, "metadata": {}})
    second = disk_run({"group": "g", "experiment_id": "b", "agents": [{"name": "ppo"}], "config": {}, "metadata": {}})
    first["backend"]["job_id"] = "job_a"
    window.runs[:0] = [first, second]
    window.jobs._logs["job_a"] = ["line 1"]
    window.select_run(first)
    open_view(window, "Logs")
    assert window.monitor.log_view.text.toPlainText() == "line 1"
    window.select_run(second)
    window.jobs._logs["job_a"].append("line 2")  # arrives while the other run is shown
    window.select_run(first)
    window.monitor.refresh_live()
    assert window.monitor.log_view.text.toPlainText().splitlines() == ["line 1", "line 2"]


# ── Runs sidebar ─────────────────────────────────────────────────────────────


def make_run(name, status, rewards=(), job_id=None):
    run = disk_run({"group": "g", "experiment_id": name, "agents": [{"name": "ppo"}], "config": {}, "metadata": {}})
    run["status"] = status
    run["metrics"] = [{"step": i, "reward": r} for i, r in enumerate(rewards)]
    if job_id:
        run["backend"]["job_id"] = job_id
    return run


def sidebar_rows(sidebar):
    tree = sidebar.tree
    return [tree.topLevelItem(i).text(0) for i in range(tree.topLevelItemCount())]


def test_sidebar_lists_live_runs_first_with_reward_trends():
    from frontend.monitor.runs_sidebar import TREND, RunsSidebar

    sidebar = RunsSidebar()
    done = make_run("done", "completed", [1.0, 5.0, 9.0])
    live = make_run("live", "running", [2.0])
    queued = make_run("waiting", "queued")
    sidebar.set_runs([done, live, queued], selected=done)
    rows = sidebar_rows(sidebar)
    assert rows[0] == "LIVE AND QUEUED" and rows[3] == "RECENT"
    assert [r.split("  ")[1] for r in rows[1:3]] == ["live", "waiting"]
    assert rows[4].startswith("✓  done")
    assert sidebar.items[done["id"]].data(1, TREND) == [1.0, 5.0, 9.0]
    done["metrics"].append({"step": 3, "reward": 12.0})
    sidebar.update_run(done)
    assert sidebar.items[done["id"]].data(1, TREND)[-1] == 12.0
    sidebar.filter.setText("wait")
    assert [run_id for run_id, item in sidebar.items.items() if not item.isHidden()] == [queued["id"]]


def test_ticking_runs_compares_them_and_survives_a_refresh():
    from frontend.monitor.runs_sidebar import PALETTE, RunsSidebar

    sidebar = RunsSidebar()
    a, b, c = (make_run(n, "completed") for n in "abc")
    sidebar.set_runs([a, b, c], selected=a)
    signals = []
    sidebar.compare_changed.connect(signals.append)
    sidebar.items[c["id"]].setCheckState(0, Qt.CheckState.Checked)
    sidebar.items[b["id"]].setCheckState(0, Qt.CheckState.Checked)
    assert signals == [[c["id"]], [c["id"], b["id"]]]  # recoloring rows did not signal again
    assert sidebar.colors() == {a["id"]: PALETTE[0], c["id"]: PALETTE[1], b["id"]: PALETTE[2]}
    sidebar.set_runs([a, c], selected=a)  # b is gone; c stays ticked
    assert sidebar.compared == [c["id"]]
    assert sidebar.items[c["id"]].checkState(0) == Qt.CheckState.Checked
    sidebar.clear_comparison()
    assert sidebar.compared == [] and signals[-1] == []


def test_the_sidebar_selects_runs_and_overlays_ticked_ones_on_the_charts(window):
    shown = make_run("shown", "running", [1.0], job_id="job_s")
    other = make_run("other", "completed", [2.0], job_id="job_o")
    window.runs[:0] = [shown, other]
    window.refresh_runs()
    window.select_run(shown)
    window.live_runs[("job_s", "ppo")] = sample_live()
    window.live_runs[("job_o", "ppo")] = sample_live()
    open_view(window, "Charts")
    chart = window.monitor.charts_view.grid.cards["eval/reward"].chart
    assert [name for name, _, _ in chart.series] == ["ppo"]
    sidebar = window.monitor.sidebar
    sidebar.items[other["id"]].setCheckState(0, Qt.CheckState.Checked)
    window.monitor.refresh_live()
    assert [(name, color) for name, _, color in chart.series] == [("shown", "#b8bb26"), ("other", "#83a598")]
    open_view(window, "System")
    system = window.monitor.system_view.grid.cards["system/cpu_percent"].chart
    assert [name for name, _, _ in system.series] == ["shown", "other"]
    sidebar.tree.setCurrentItem(sidebar.items[other["id"]])  # clicking a row shows that run
    assert window.selected is other
    assert window.monitor.run_combo.itemData(window.monitor.run_combo.currentIndex()) == other["id"]


# ── Point inspector ──────────────────────────────────────────────────────────


def test_nearest_picks_the_closest_logged_step():
    from frontend.monitor.live_view import nearest

    points = [(1000.0, 1.0), (2000.0, 2.0), (4000.0, 4.0)]
    assert nearest(points, 2000.0) == (2000.0, 2.0)
    assert nearest(points, 2900.0) == (2000.0, 2.0)
    assert nearest(points, 3100.0) == (4000.0, 4.0)
    assert nearest(points, -5.0) == (1000.0, 1.0) and nearest(points, 9e9) == (4000.0, 4.0)
    assert nearest([], 1.0) is None


def test_hovering_a_chart_lists_every_metric_of_every_series():
    view = ChartsView()
    sparse = LiveRun()  # evaluations and losses on separate rows, as Lightning logs them
    sparse.add(
        [
            record("metrics", step=1, data={"eval/reward": 50.0, "transitions": 2000.0}),
            record("metrics", step=2, data={"losses/total_loss": 0.4, "transitions": 2048.0}),
        ],
        now=T0,
    )
    view.show_lives({"ppo": sample_live(), "dqn": sparse}, {"ppo": "#b8bb26", "dqn": "#83a598"})
    view.grid.cards["eval/reward"].chart.point_hovered.emit(2000.0)
    table = view.inspector.table
    assert [table.horizontalHeaderItem(c).text() for c in range(table.columnCount())] == ["ppo", "dqn"]
    assert table.horizontalHeaderItem(1).foreground().color().name() == "#83a598"
    rows = [table.verticalHeaderItem(r).text() for r in range(table.rowCount())]
    assert rows == ["eval/reward", "losses/total_loss"]  # evaluation before losses, like the sections
    cell = {
        (rows[r], table.horizontalHeaderItem(c).text()): table.item(r, c) for r in range(len(rows)) for c in range(2)
    }
    assert cell[("eval/reward", "ppo")].text() == "30.0"  # logged at exactly 2,000
    assert cell[("eval/reward", "dqn")].text() == "50.0"
    assert cell[("losses/total_loss", "dqn")].text().startswith("≈ ")  # nearest: 2,048
    assert "2,048" in cell[("losses/total_loss", "dqn")].toolTip()
    assert "environment steps = 2,000" in view.inspector.caption.text() and "≈ marks 1" in view.inspector.caption.text()


def test_the_inspected_step_stays_current_as_values_arrive():
    view = ChartsView()
    live = LiveRun()
    live.add([record("metrics", step=1, data={"eval/reward": 5.0, "transitions": 100.0})], now=T0)
    view.show_lives({"ppo": live})
    view.grid.point_hovered.emit(100.0)
    live.add([record("metrics", step=2, data={"val/f1": 0.7, "transitions": 100.0})], now=T0 + 1)
    view.show_lives({"ppo": live})
    table = view.inspector.table
    assert [table.verticalHeaderItem(r).text() for r in range(table.rowCount())] == ["eval/reward", "val/f1"]


def test_the_overview_scrolls_rather_than_squashing_its_charts(window):
    from frontend.monitor.panel import OVERVIEW_REWARD_HEIGHT

    monitor = window.monitor
    assert monitor.views.widget(0) is monitor.overview_scroll
    assert monitor.overview_scroll.verticalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAsNeeded
    window.resize(1200, 500)  # too short for the cards and both charts
    window.tabs.setCurrentWidget(window.monitor_panel)
    window.show()
    for _ in range(5):
        app.processEvents()
    assert monitor.reward_chart.height() >= OVERVIEW_REWARD_HEIGHT
    assert monitor.overview_scroll.verticalScrollBar().maximum() > 0

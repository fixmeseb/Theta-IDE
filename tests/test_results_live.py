"""Results panel, phase 6: following runs that are still training, and staying fast at scale."""

import os
import sys
from urllib.parse import parse_qs, urlparse

import pytest

pytest.importorskip("PyQt6")

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PyQt6.QtCore import Qt  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)
app = QApplication.instance() or QApplication(sys.argv[:1])

from frontend.results.data import MetricTable, Source, thin  # noqa: E402
from frontend.results.loader import ResultsLoader  # noqa: E402
from frontend.results.panel import RELIST_EVERY, ResultsPanel  # noqa: E402

SOURCE = Source("g", "live_run", "ppo")


class LiveBackend:
    """A run whose metrics.csv grows between requests, served like GET …/metrics?since=N."""

    def __init__(self):
        self.rows = [{"transitions": "1000", "eval/reward": "10"}]
        self.version = "version_0"
        self.finished = False
        self.requests = []

    def get(self, path, callback):
        self.requests.append(path)
        url = urlparse(path)
        if url.path == "/api/runs":
            agents = [{"name": "ppo"}]
            callback(
                {"runs": [{"group": "g", "experiment_id": "live_run", "agents": agents, "finished": self.finished}]},
                None,
            )
        elif url.path.endswith("/manifest"):
            callback(
                {"group": "g", "experiment_id": "live_run", "agents": {"ppo": [self.version]}, "artifacts": []}, None
            )
        elif url.path.endswith("/metrics"):
            since = int(parse_qs(url.query).get("since", ["0"])[0])
            callback(
                {
                    "metrics": [dict(r) for r in self.rows[since:]],
                    "columns": ["transitions", "eval/reward"],
                    "total_rows": len(self.rows),
                    "version": self.version,
                },
                None,
            )
        else:
            callback(None, f"HTTP 404: {path}")

    def get_bytes(self, path, callback):
        callback(None, "HTTP 404")


def metrics_requests(backend):
    return [p for p in backend.requests if "/metrics" in p]


# ── Loader ───────────────────────────────────────────────────────────────────


def test_poll_appends_only_new_rows():
    backend = LiveBackend()
    loader = ResultsLoader(backend)
    loader.load_table(SOURCE)
    backend.rows.append({"transitions": "2000", "eval/reward": "30"})
    loader.poll_table(SOURCE)
    assert metrics_requests(backend)[-1].endswith("/metrics")  # since=0 (the default): the last row held is re-read
    assert loader.tables[SOURCE].series("eval/reward") == [(1000.0, 10.0), (2000.0, 30.0)]
    backend.rows.append({"transitions": "3000", "eval/reward": "45"})
    loader.poll_table(SOURCE)
    assert metrics_requests(backend)[-1].endswith("metrics?since=1")
    assert loader.tables[SOURCE].row_count == 3


def test_poll_repairs_a_row_that_was_read_half_written():
    backend = LiveBackend()
    backend.rows = [{"transitions": "1000", "eval/reward": "10"}, {"transitions": "2000", "eval/reward": ""}]
    loader = ResultsLoader(backend)
    loader.load_table(SOURCE)  # caught mid-write: the second row has no reward yet
    assert loader.tables[SOURCE].series("eval/reward") == [(1000.0, 10.0)]
    backend.rows[1] = {"transitions": "2000", "eval/reward": "30"}
    loaded = []
    loader.table_loaded.connect(loaded.append)
    loader.poll_table(SOURCE)
    assert loader.tables[SOURCE].series("eval/reward") == [(1000.0, 10.0), (2000.0, 30.0)]
    assert loaded == [SOURCE] and loader.tables[SOURCE].row_count == 2
    loader.poll_table(SOURCE)  # nothing new: no redraw
    assert loaded == [SOURCE]


def test_poll_refetches_when_a_new_version_starts():
    backend = LiveBackend()
    loader = ResultsLoader(backend)
    loader.load_table(SOURCE)
    backend.version, backend.rows = "version_1", [{"transitions": "500", "eval/reward": "99"}]
    loader.poll_table(SOURCE)
    assert loader.table_versions[SOURCE] == "version_1"
    assert loader.tables[SOURCE].series("eval/reward") == [(500.0, 99.0)]  # not mixed with version_0


# ── Panel ────────────────────────────────────────────────────────────────────


def live_panel():
    backend = LiveBackend()
    panel = ResultsPanel(backend)
    panel.poll_timer.stop()  # ticks are driven by hand here
    panel.refresh()
    panel.explorer.tree.topLevelItem(0).child(0).setCheckState(0, Qt.CheckState.Checked)
    return panel, backend


def test_panel_follows_ticked_runs_still_training():
    panel, backend = live_panel()
    assert panel.live_sources() == [SOURCE] and "following 1 run still training" in panel.status.text()
    backend.rows.append({"transitions": "2000", "eval/reward": "30"})
    panel.poll()  # hidden panels do not poll
    assert len(panel.grid.cards["eval/reward"].chart.series[0][1]) == 1
    panel.show()
    panel.poll()
    assert [p["eval/reward"] for p in panel.grid.cards["eval/reward"].chart.series[0][1]] == [10.0, 30.0]
    assert panel.summary.rows[0][2] == 30.0  # the Summary follows too


def test_panel_relists_runs_and_stops_following_finished_ones():
    panel, backend = live_panel()
    panel.show()
    tree_before = panel.explorer.tree.topLevelItem(0)
    backend.finished = True
    for _ in range(RELIST_EVERY):
        panel.poll()
    assert panel.unfinished == set() and panel.live_sources() == []
    assert "following" not in panel.status.text()
    # The run list itself did not change, so the tree (and what is ticked) was left alone.
    assert panel.explorer.tree.topLevelItem(0) is tree_before
    assert panel.sources == [SOURCE]


# ── Scale ────────────────────────────────────────────────────────────────────


def test_thin_keeps_spikes_and_the_latest_point():
    points = [{"step": i, "m": 0.0} for i in range(10_000)]
    points[4321]["m"] = 99.0  # a spike
    points[777]["m"] = -5.0  # a dip
    drawn = thin(points, "m", limit=200)
    assert len(drawn) <= 202
    assert {p["m"] for p in drawn} >= {99.0, -5.0, 0.0}
    assert drawn[-1] is points[-1] and [p["step"] for p in drawn] == sorted(p["step"] for p in drawn)
    short = points[:50]
    assert thin(short, "m", limit=200) is short


def test_long_series_are_thinned_for_drawing_only(tmp_path):
    backend = LiveBackend()
    backend.finished = True
    backend.rows = [{"transitions": str(i), "eval/reward": str(i % 7)} for i in range(6000)]
    panel = ResultsPanel(backend)
    panel.poll_timer.stop()
    panel.refresh()
    panel.explorer.tree.topLevelItem(0).child(0).setCheckState(0, Qt.CheckState.Checked)
    drawn = panel.grid.cards["eval/reward"].chart.series[0][1]
    assert len(drawn) <= 2001
    assert panel.summary.rows[0][-1] == 6000  # Summary uses every point
    assert len(panel.curves_csv().splitlines()) == 6001  # so does the export (header + every point)


def test_metric_table_truncate_and_row():
    table = MetricTable.from_rows([{"x": "1", "a": "2"}, {"x": "2", "b": "5"}])
    assert table.row(1) == {"x": 2.0, "b": 5.0}
    table.truncate(1)
    assert table.row_count == 1 and "b" not in table.columns and table.row(0) == {"x": 1.0, "a": 2.0}


def test_backend_run_summaries_are_cached_until_files_change(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import src.app.api.app as api

    monkeypatch.chdir(tmp_path)
    version = tmp_path / "results" / "logs" / "g" / "exp" / "ppo" / "version_0"
    version.mkdir(parents=True)
    metrics = version / "metrics.csv"
    metrics.write_text("transitions,eval/reward\n1000,10\n", encoding="utf-8")
    reads = []
    real = api._read_agent_summary
    monkeypatch.setattr(api, "_read_agent_summary", lambda *a: reads.append(a[0]) or real(*a))
    client = TestClient(api.app)
    first = client.get("/api/runs").json()["runs"][0]["agents"][0]
    client.get("/api/runs")
    assert len(reads) == 1  # unchanged files: served from the cache
    metrics.write_text("transitions,eval/reward\n1000,10\n2000,50\n", encoding="utf-8")  # training went on
    second = client.get("/api/runs").json()["runs"][0]["agents"][0]
    assert len(reads) == 2 and (first["best_reward"], second["best_reward"]) == (10.0, 50.0)

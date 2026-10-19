"""Plot viewer: it draws exactly the runs ticked in the runs list, or a file opened from disk."""

import os
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PyQt6.QtWidgets")
app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])

from frontend.plots import PlotViewer  # noqa: E402


def run(run_id, reward, extra=None):
    return {"id": run_id, "simulated": True, "config": {"name": run_id},
            "metrics": [{"step": s, "reward": reward + s, **(extra or {})} for s in (1, 2, 3)]}


@pytest.fixture
def viewer():
    viewer = PlotViewer()
    viewer.update_runs([run("a", 0), run("b", 10), run("c", 20, {"loss": 0.5})])
    yield viewer
    viewer.deleteLater()


def drawn(viewer):
    return [(name.split(" [")[0], color) for name, _, color in viewer.canvas.series]


def test_there_is_no_source_picker_and_nothing_is_drawn_until_runs_are_ticked(viewer):
    assert not hasattr(viewer, "source") and not hasattr(viewer, "compare")
    assert drawn(viewer) == []
    assert "Tick runs" in viewer.canvas.empty_text


def test_ticked_runs_are_drawn_in_tick_order_and_row_colors(viewer):
    viewer.set_compared(["c", "a"], {"c": "#b8bb26", "a": "#83a598"})
    assert drawn(viewer) == [("c", "#b8bb26"), ("a", "#83a598")]
    assert viewer.caption.text().startswith("2 runs")
    viewer.set_compared([], {})
    assert drawn(viewer) == []


def test_the_axes_offer_every_column_of_the_ticked_runs(viewer):
    viewer.set_compared(["a"], {"a": "#b8bb26"})
    assert [viewer.yaxis.itemText(i) for i in range(viewer.yaxis.count())] == ["step", "reward"]
    viewer.set_compared(["a", "c"], {"a": "#b8bb26", "c": "#83a598"})
    assert "loss" in [viewer.yaxis.itemText(i) for i in range(viewer.yaxis.count())]
    viewer.yaxis.setCurrentText("loss")  # only c logs it: a is listed as unavailable
    assert drawn(viewer) == [("c", "#83a598")] and "chosen axes unavailable" in viewer.caption.text()


def test_runs_that_are_gone_are_skipped(viewer):
    viewer.set_compared(["gone", "b"], {"gone": "#d3869b", "b": "#fabd2f"})
    assert drawn(viewer) == [("b", "#fabd2f")]


def test_an_opened_csv_is_shown_until_the_ticks_change(viewer, tmp_path):
    viewer.set_compared(["a"], {"a": "#b8bb26"})
    path = tmp_path / "metrics.csv"
    path.write_text("step,reward\n1,5\n2,6\n", encoding="utf-8")
    viewer.open_path(path)
    assert drawn(viewer) == [("metrics.csv", "#b8bb26")]
    viewer.update_runs(viewer.runs)  # new metrics arriving do not take the file away
    assert drawn(viewer) == [("metrics.csv", "#b8bb26")]
    viewer.set_compared(["a", "b"], {"a": "#b8bb26", "b": "#83a598"})
    assert drawn(viewer) == [("a", "#b8bb26"), ("b", "#83a598")]


def test_a_long_legend_still_paints(viewer):
    many = [run(f"run_with_a_long_name_{i}", i) for i in range(12)]
    viewer.update_runs(many)
    viewer.set_compared([r["id"] for r in many], {r["id"]: "#83a598" for r in many})
    viewer.resize(700, 500)
    assert len(viewer.canvas.series) == 12
    assert not viewer.canvas.grab().isNull()

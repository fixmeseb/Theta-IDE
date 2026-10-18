"""Tests for the figure workspace.

The spec and rendering run headless, since model.py imports no Qt. The view
tests cover what is easy to get wrong in a composition UI: that shrinking
the grid says what it threw away, that the preview never inherits the export
DPI, and that an unavailable panel leaves a marked Axes instead of silently
changing the layout.
"""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import pytest
from matplotlib.figure import Figure

from frontend.figures.model import (
    MAX_SLOTS,
    FigureSpec,
    PanelRef,
    available_experiments,
    render,
)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt6.QtWidgets import QApplication

    _app = QApplication.instance() or QApplication(sys.argv[:1])
    from frontend.figures.view import PREVIEW_DPI, FigureWorkspace

    HAS_QT = True
except ImportError:  # pragma: no cover
    HAS_QT = False


def ref(key="p/m", label="Panel"):
    return PanelRef(key=key, exp_id="exp", plotter="p", label=label)


class FakePanel:
    """Stands in for plot.base.PanelSpec."""

    def __init__(self, drew=True, boom=False):
        self.drew, self.boom = drew, boom

    def draw(self, ax):
        if self.boom:
            raise RuntimeError("plotter exploded")
        if self.drew:
            ax.plot([1, 2, 3], [1, 4, 9], label="series")
        return self.drew


def resolver(panel=None):
    return lambda _ref: panel if panel is not None else FakePanel()


# ── The spec ──────────────────────────────────────────────────────────────


def test_a_new_spec_has_empty_slots():
    spec = FigureSpec(rows=2, cols=2)
    assert spec.slot_count == 4
    assert spec.panels == [None] * 4
    assert spec.used() == []


def test_panels_fill_the_first_free_slot():
    spec = FigureSpec(rows=1, cols=2)
    assert spec.add(ref()) == 0
    assert spec.add(ref()) == 1


def test_adding_beyond_the_grid_grows_a_row():
    spec = FigureSpec(rows=1, cols=2)
    for _ in range(3):
        spec.add(ref())
    assert (spec.rows, spec.cols) == (2, 2)
    assert len(spec.used()) == 3


def test_a_figure_is_capped():
    spec = FigureSpec(rows=1, cols=1)
    with pytest.raises(ValueError, match="limited to"):
        for _ in range(MAX_SLOTS + 2):
            spec.add(ref())


def test_shrinking_reports_what_no_longer_fits():
    spec = FigureSpec(rows=2, cols=2)
    for i in range(4):
        spec.add(ref(label=f"P{i}"))
    dropped = spec.set_grid(1, 2)
    assert [d.label for d in dropped] == ["P2", "P3"]
    assert len(spec.used()) == 2


def test_shrinking_drops_empty_slots_before_real_panels():
    spec = FigureSpec(rows=2, cols=2)
    spec.add(ref(label="keep"))
    assert spec.set_grid(1, 1) == []
    assert [p.label for p in spec.used()] == ["keep"]


def test_remove_leaves_a_hole_that_compact_closes():
    spec = FigureSpec(rows=1, cols=3)
    for i in range(3):
        spec.add(ref(label=f"P{i}"))
    spec.remove(0)
    assert spec.panels[0] is None
    spec.compact()
    assert [p.label for p in spec.used()] == ["P1", "P2"]


def test_move_swaps_two_slots():
    spec = FigureSpec(rows=1, cols=2)
    spec.add(ref(label="a"))
    spec.add(ref(label="b"))
    spec.move(0, 1)
    assert [p.label for p in spec.panels] == ["b", "a"]


def test_out_of_range_edits_are_ignored():
    spec = FigureSpec(rows=1, cols=1)
    spec.add(ref())
    spec.move(0, 9)
    spec.update(9, title="x")
    assert spec.remove(9) is None
    assert len(spec.used()) == 1


def test_update_restyles_in_place():
    spec = FigureSpec(rows=1, cols=1)
    spec.add(ref())
    spec.update(0, title="New", color="#b8bb26")
    assert spec.panels[0].title == "New"
    assert spec.panels[0].color == "#b8bb26"


# ── Rendering ─────────────────────────────────────────────────────────────


def test_render_makes_one_axes_per_filled_slot():
    spec = FigureSpec(rows=2, cols=2)
    spec.add(ref())
    spec.add(ref())
    fig = render(spec, resolver(), figure=Figure())
    assert len(fig.axes) == 2


def test_render_applies_the_overrides():
    spec = FigureSpec(rows=1, cols=1)
    spec.add(ref())
    spec.update(0, title="Chosen", color="#ff0000", line_width=3.5)
    ax = render(spec, resolver(), figure=Figure()).axes[0]
    assert ax.get_title() == "Chosen"
    assert ax.lines[0].get_linewidth() == 3.5


def test_an_unfilled_title_template_falls_back_to_the_label():
    # plot/base.py never supplies {display_name}, so the raw template arrives.
    class Templated(FakePanel):
        def draw(self, ax):
            ax.plot([1, 2], [1, 2])
            ax.set_title("{display_name}: {clean_metric}")
            return True

    spec = FigureSpec(rows=1, cols=1)
    spec.add(ref(label="losses: Total Loss"))
    ax = render(spec, resolver(Templated()), figure=Figure()).axes[0]
    assert ax.get_title() == "losses: Total Loss"


def test_an_explicit_title_still_wins_over_a_template():
    spec = FigureSpec(rows=1, cols=1)
    spec.add(ref())
    spec.update(0, title="Mine")
    ax = render(spec, resolver(), figure=Figure()).axes[0]
    assert ax.get_title() == "Mine"


def test_a_missing_source_leaves_a_marked_axes():
    spec = FigureSpec(rows=1, cols=2)
    spec.add(ref(label="Gone"))
    spec.add(ref())
    fig = render(spec, lambda r: None if r.label == "Gone" else FakePanel(), figure=Figure())
    # The layout must not shift when one source disappears.
    assert len(fig.axes) == 2
    assert any("unavailable" in t.get_text() for t in fig.axes[0].texts)


def test_a_plotter_that_raises_does_not_lose_the_figure():
    spec = FigureSpec(rows=1, cols=2)
    spec.add(ref(label="Boom"))
    spec.add(ref())
    fig = render(spec, lambda r: FakePanel(boom=r.label == "Boom"), figure=Figure())
    assert len(fig.axes) == 2
    assert any("failed to draw" in t.get_text() for t in fig.axes[0].texts)


def test_a_panel_with_no_data_is_marked():
    spec = FigureSpec(rows=1, cols=1)
    spec.add(ref())
    fig = render(spec, resolver(FakePanel(drew=False)), figure=Figure())
    assert any("no data" in t.get_text() for t in fig.axes[0].texts)


def test_render_reuses_the_figure_it_is_given():
    spec = FigureSpec(rows=1, cols=1)
    spec.add(ref())
    fig = Figure()
    assert render(spec, resolver(), figure=fig) is fig
    render(spec, resolver(), figure=fig)
    assert len(fig.axes) == 1, "a redraw must clear, not accumulate axes"


def test_render_leaves_dpi_alone_unless_asked():
    spec = FigureSpec(rows=1, cols=1, dpi=300)
    spec.add(ref())
    fig = Figure(dpi=100)
    render(spec, resolver(), figure=fig)
    assert fig.get_dpi() == 100, "the preview must not inherit the export DPI"
    render(spec, resolver(), figure=fig, dpi=300)
    assert fig.get_dpi() == 300


def test_suptitle_is_set_when_given():
    spec = FigureSpec(rows=1, cols=1, suptitle="Figure 1")
    spec.add(ref())
    assert render(spec, resolver(), figure=Figure())._suptitle.get_text() == "Figure 1"


def test_available_experiments_handles_a_missing_root(tmp_path):
    assert available_experiments(tmp_path / "nope") == []


def test_available_experiments_finds_runs_with_metrics(tmp_path):
    run = tmp_path / "grp" / "exp" / "method" / "version_0"
    run.mkdir(parents=True)
    (run / "metrics.csv").write_text("epoch\n0\n")
    (tmp_path / "grp" / "empty_exp").mkdir()
    assert available_experiments(tmp_path) == [("grp", "exp")]


# ── The widget ────────────────────────────────────────────────────────────


@pytest.mark.skipif(not HAS_QT, reason="PyQt6 not installed")
class TestFigureWorkspace:
    def test_preview_renders_at_screen_dpi_not_export_dpi(self):
        w = FigureWorkspace()
        assert w.figure.get_dpi() == PREVIEW_DPI
        w.spn_dpi.setValue(600)
        assert w.spec.dpi == 600
        assert w.figure.get_dpi() == PREVIEW_DPI

    def test_status_prompts_when_the_figure_is_empty(self):
        assert "Add a panel" in FigureWorkspace().lbl_status.text()

    def test_adding_without_a_selection_explains_itself(self):
        w = FigureWorkspace()
        w.lst_catalog.setCurrentRow(-1)
        w._add_selected()
        assert "Pick a panel" in w.lbl_status.text()

    def test_shrinking_the_grid_says_what_it_removed(self):
        w = FigureWorkspace()
        for i in range(4):
            w.spec.add(ref(label=f"P{i}"))
        w._sync_grid_boxes()
        w._refresh_placed()
        w.spn_rows.setValue(1)
        w.spn_cols.setValue(2)
        assert "no longer fit" in w.lbl_status.text()
        assert "P" in w.lbl_status.text()

    def test_export_refuses_an_empty_figure(self):
        w = FigureWorkspace()
        w._export()
        assert "nothing on the figure" in w.lbl_status.text()

    def test_figure_settings_reach_the_spec(self):
        w = FigureWorkspace()
        w.spn_width.setValue(7.5)
        w.txt_suptitle.setText("Figure 2")
        w.chk_tight.setChecked(False)
        assert w.spec.width == 7.5
        assert w.spec.suptitle == "Figure 2"
        assert w.spec.tight is False

    def test_renaming_a_panel_updates_the_placed_list(self):
        w = FigureWorkspace()
        w.spec.add(ref(label="Original"))
        w._refresh_placed()
        w.lst_placed.setCurrentRow(0)
        w._update(0, title="Renamed")
        assert "Renamed" in w.lst_placed.item(0).text()

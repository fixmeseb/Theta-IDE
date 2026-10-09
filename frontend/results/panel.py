"""Results panel: tick runs on the left; see every metric, value and file they produced on the right."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QMenu,
    QPushButton,
    QSlider,
    QSplitter,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..widgets import label
from .artifacts import ArtifactBrowser
from .config_diff import ConfigDiff, run_config_path
from .data import (
    COMBINE_MODES,
    SPREAD,
    Source,
    aggregate,
    chart_points,
    combined_points,
    series_groups,
    smooth,
    summary_rows,
    to_csv,
)
from .explorer import RunExplorer
from .inspector import PointInspector
from .loader import ResultsLoader
from .metrics_catalog import AXIS_COLUMNS
from .plot_grid import PlotGrid, chart_title
from .summary import SummaryView

PALETTE = (
    "#b8bb26",
    "#83a598",
    "#fabd2f",
    "#d3869b",
    "#8ec07c",
    "#fe8019",
    "#fb4934",
    "#689d6a",
    "#458588",
    "#d79921",
    "#b16286",
    "#98971a",
)
# Artifact tabs: the kinds each lists, and what it says when the ticked runs have none.
ARTIFACT_TABS = {
    "Tables": (("table", "metrics"), "The ticked runs have no tables."),
    "Figures": (("figure",), "The ticked runs have no figures yet."),
    "Reports": (("report",), "The ticked runs have no reports yet."),
    "Config": (("config", "metadata"), "The ticked runs have no config files."),
    "Files": (None, "Tick runs to list every file they produced."),
}
AXIS_LABELS = {
    "transitions": "Environment steps",
    "step": "Logged step",
    "epoch": "Epoch",
    "time/total": "Wall time (s)",
    "training_time_seconds": "Training time (s)",
}


class ResultsPanel(QWidget):
    """Everything the ticked runs produced.

    Curves: one chart per metric, with a point inspector showing every value at the hovered step;
    an agent's versions (repeated runs) can be combined into a mean ± std or ± SEM band.
    Summary: final, best, range, mean and area under the curve per series and metric, plus when a
    threshold was first reached. Export writes the visible curves or the summary as CSV, or the charts as PNG.
    Tables, Figures, Reports, Config (with a cross-run diff) and Files: every artifact in the runs'
    manifests, previewed by kind.

    `backend` needs get(path, callback); `settings` (optional) needs get(*keys, default=) and
    set(*keys, value), and keeps hidden plots across sessions under [results] hidden_plots.
    `load_config(run_key)` (optional) puts a run's saved builder config into the experiment builder and
    returns None, or returns why it could not.
    """

    def __init__(self, backend, settings=None, load_config=None, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.load_config = load_config
        self.curves: dict[str, list] = {}  # metric -> [(label, chart points, color, combined)], as drawn
        self.load_button = None
        self.loader = ResultsLoader(backend, self)
        self.sources: list[Source] = []
        self.hidden: set[str] = set(self._setting("hidden_plots", []))
        self.smoothing = 0.0

        root = QVBoxLayout(self)
        root.setContentsMargins(18, 14, 18, 12)
        root.setSpacing(10)

        toolbar = QHBoxLayout()
        toolbar.addWidget(label("RESULTS / EVERY METRIC", "eyebrow"))
        toolbar.addStretch()
        self.plots_button = QToolButton()
        self.plots_button.setText("Plots ▾")
        self.plots_button.setToolTip("Show or hide individual plots")
        self.plots_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.plots_menu = QMenu(self.plots_button)
        self.plots_menu.aboutToShow.connect(self._fill_plots_menu)
        self.plots_button.setMenu(self.plots_menu)
        toolbar.addWidget(self.plots_button)
        toolbar.addWidget(label("X-axis", "muted"))
        self.axis_select = QComboBox()
        self.axis_select.setToolTip("What the horizontal axis of every chart measures")
        self.axis_select.currentIndexChanged.connect(lambda _: self.redraw())
        toolbar.addWidget(self.axis_select)
        self.smoothing_label = label("Smoothing: 0%", "muted")
        toolbar.addWidget(self.smoothing_label)
        self.smoothing_slider = QSlider(Qt.Orientation.Horizontal)
        self.smoothing_slider.setRange(0, 95)
        self.smoothing_slider.setFixedWidth(110)
        self.smoothing_slider.setToolTip("Exponential moving average smoothing (0% = raw data)")
        self.smoothing_slider.valueChanged.connect(self._smoothing_changed)
        toolbar.addWidget(self.smoothing_slider)
        toolbar.addWidget(label("Versions", "muted"))
        self.combine_select = QComboBox()
        for mode, text in COMBINE_MODES.items():
            self.combine_select.addItem(text, mode)
        self.combine_select.setToolTip(
            "Combine the ticked versions (repeated runs, e.g. seeds) of each agent into one mean curve with a band"
        )
        self.combine_select.currentIndexChanged.connect(lambda _: self.redraw())
        toolbar.addWidget(self.combine_select)
        self.export_button = QToolButton()
        self.export_button.setText("Export ▾")
        self.export_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        export_menu = QMenu(self.export_button)
        export_menu.addAction("Visible curves as CSV…", lambda: self._export("curves"))
        export_menu.addAction("Summary as CSV…", lambda: self._export("summary"))
        export_menu.addAction("Charts as PNG…", lambda: self._export("png"))
        self.export_button.setMenu(export_menu)
        toolbar.addWidget(self.export_button)
        refresh = QPushButton("↻  Refresh")
        refresh.setToolTip("Re-read results/ through the backend")
        refresh.clicked.connect(self.refresh)
        toolbar.addWidget(refresh)
        root.addLayout(toolbar)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.explorer = RunExplorer()
        self.explorer.setMinimumWidth(220)
        self.explorer.selection_changed.connect(self._selection_changed)
        self.explorer.manifest_wanted.connect(self.loader.load_manifest)
        splitter.addWidget(self.explorer)
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        curves = QWidget()
        curves_layout = QVBoxLayout(curves)
        curves_layout.setContentsMargins(8, 6, 0, 0)
        curves_layout.setSpacing(8)
        self.legend = label("", "muted")
        self.legend.setWordWrap(True)
        self.legend.setTextFormat(Qt.TextFormat.RichText)
        curves_layout.addWidget(self.legend)
        curves_split = QSplitter(Qt.Orientation.Vertical)
        self.grid = PlotGrid()
        self.grid.hide_requested.connect(lambda metric: self.set_hidden(metric, True))
        self.grid.point_hovered.connect(self._point_hovered)
        curves_split.addWidget(self.grid)
        self.inspector = PointInspector()
        curves_split.addWidget(self.inspector)
        curves_split.setSizes([620, 200])
        curves_layout.addWidget(curves_split, 1)
        self.tabs.addTab(curves, "Curves")
        self.summary = SummaryView()
        self.summary.changed.connect(self.update_summary)
        self.tabs.addTab(self.summary, "Summary")
        # Artifact tabs start empty and build their browser when first opened: each one is hundreds of
        # widgets, and the app restyles every widget when its theme is applied.
        self.browsers: dict[str, ArtifactBrowser] = {}
        self.config_diff: ConfigDiff | None = None
        self._pages: dict[str, QWidget] = {}
        for name in ARTIFACT_TABS:
            page = QWidget()
            QVBoxLayout(page).setContentsMargins(0, 0, 0, 0)
            self._pages[name] = page
            self.tabs.addTab(page, name)
        self.tabs.currentChanged.connect(self._tab_opened)
        splitter.addWidget(self.tabs)
        splitter.setSizes([260, 900])
        root.addWidget(splitter, 1)
        self.status = label("", "muted")
        root.addWidget(self.status)

        self.loader.runs_listed.connect(self._runs_listed)
        self.loader.manifest_loaded.connect(self.explorer.add_manifest)
        self.loader.manifest_loaded.connect(lambda _manifest: self.update_artifacts())
        self.loader.file_loaded.connect(self._file_loaded)
        self.loader.table_loaded.connect(lambda _source: self.redraw())
        self.loader.failed.connect(lambda message: self.status.setText(f"Could not load: {message}"))
        self.redraw()

    # ── Settings ─────────────────────────────────────────────────────────────

    def _setting(self, key, default):
        if self.settings is None:
            return default
        value = self.settings.get("results", key, default=default)
        return value if isinstance(value, type(default)) else default

    def set_hidden(self, metric: str, hidden: bool):
        (self.hidden.add if hidden else self.hidden.discard)(metric)
        if self.settings is not None:
            self.settings.set("results", "hidden_plots", sorted(self.hidden))
        self.redraw()

    def show_all(self):
        self.hidden.clear()
        if self.settings is not None:
            self.settings.set("results", "hidden_plots", [])
        self.redraw()

    # ── Data flow ────────────────────────────────────────────────────────────

    def refresh(self):
        """Re-list runs and drop cached data, so new and still-training runs are re-read."""
        self.loader.clear()
        self.status.setText("Listing runs…")
        self.loader.list_runs()

    def set_offline(self):
        self.status.setText(
            "Backend offline: start it to browse results (uvicorn src.app.api.app:app --host 127.0.0.1 --port 8000)."
        )

    def _runs_listed(self, entries):
        self.explorer.set_runs(entries)
        count = sum(1 for e in entries if e.get("agents"))
        self.status.setText(
            f"{count} run{'s' * (count != 1)} in results/logs. Tick runs to plot every metric they logged."
        )

    def _selection_changed(self, sources):
        self.sources = sources
        for source in sources:
            self.loader.load_table(source)
            self.loader.load_manifest(source.group, source.experiment_id)
        self.redraw()
        self.update_artifacts()

    def run_keys(self) -> list[str]:
        """Ticked runs, in tree order, each once (several agents or versions share a run)."""
        return list(dict.fromkeys(source.run_key for source in self.sources))

    # ── Artifacts ────────────────────────────────────────────────────────────

    def browser(self, name: str) -> ArtifactBrowser:
        """The artifact browser of one tab ("Tables", "Figures", …), built on first use."""
        if name not in self.browsers:
            kinds, empty = ARTIFACT_TABS[name]
            browser = self.browsers[name] = ArtifactBrowser(self.loader, kinds, empty)
            layout = self._pages[name].layout()
            if name == "Config":
                self.config_diff = ConfigDiff()
                top = QWidget()
                top_layout = QVBoxLayout(top)
                top_layout.setContentsMargins(0, 0, 0, 0)
                actions = QHBoxLayout()
                actions.addStretch()
                self.load_button = QPushButton("Load in experiment builder")
                self.load_button.setToolTip("Put the ticked run's saved builder settings into the Experiment tab")
                self.load_button.clicked.connect(self.load_in_builder)
                actions.addWidget(self.load_button)
                top_layout.addLayout(actions)
                top_layout.addWidget(self.config_diff, 1)
                split = QSplitter(Qt.Orientation.Vertical)
                split.addWidget(top)
                split.addWidget(browser)
                split.setSizes([300, 400])
                layout.addWidget(split)
            else:
                layout.addWidget(browser)
            self.update_artifacts()
        return self.browsers[name]

    def _tab_opened(self, index):
        name = self.tabs.tabText(index)
        if name in ARTIFACT_TABS:
            self.browser(name)

    def manifests(self):
        """Loaded manifests of the ticked runs."""
        return [self.loader.manifests[k] for k in self.run_keys() if k in self.loader.manifests]

    def load_in_builder(self):
        """Load the one ticked run's saved builder config into the experiment builder."""
        runs = self.run_keys()
        if len(runs) != 1:
            self.status.setText("Tick exactly one run to load its saved config into the experiment builder.")
            return
        problem = self.load_config(runs[0]) if self.load_config else "The experiment builder is not available here."
        self.status.setText(problem or f"Loaded the saved config of {runs[0]} into the experiment builder.")

    def update_artifacts(self):
        if self.load_button is not None:
            self.load_button.setEnabled(len(self.run_keys()) == 1)
        manifests = self.manifests()
        for browser in self.browsers.values():
            browser.set_manifests(manifests)
        if self.config_diff is None:
            return
        for manifest in manifests:
            path = run_config_path(manifest.run_key)
            if any(a["path"] == path for a in manifest.artifacts):
                self.loader.load_file(manifest.run_key, path)
        self._show_config_diff()

    def _file_loaded(self, run_key, path):
        if path == run_config_path(run_key):
            self._show_config_diff()

    def _show_config_diff(self):
        if self.config_diff is None:
            return
        configs, missing = {}, []
        for key in self.run_keys():
            manifest = self.loader.manifests.get(key)
            if manifest is None:
                continue  # still loading
            data = self.loader.files.get((key, run_config_path(key)))
            if data is not None:
                configs[key] = data
            elif not any(a["path"] == run_config_path(key) for a in manifest.artifacts):
                missing.append(key)
        self.config_diff.show_configs(configs, missing)

    # ── Point inspector ──────────────────────────────────────────────────────

    def _point_hovered(self, x):
        axis = self.axis_select.currentData()
        if axis:
            self.inspector.show_point(axis, x, self.loaded_tables())

    def _smoothing_changed(self, value):
        self.smoothing = value / 100.0
        self.smoothing_label.setText(f"Smoothing: {value}%")
        self.redraw()

    # ── Drawing ──────────────────────────────────────────────────────────────

    def loaded_tables(self):
        return [(s, self.loader.tables[s]) for s in self.sources if s in self.loader.tables]

    def all_metrics(self) -> list[str]:
        """Every metric any ticked run logged, including hidden ones."""
        seen: dict[str, None] = {}
        for _, table in self.loaded_tables():
            for metric in table.metrics():
                seen.setdefault(metric)
        return list(seen)

    def _sync_axes(self, tables):
        available = [axis for axis in AXIS_COLUMNS if any(axis in t.columns for _, t in tables)]
        current = self.axis_select.currentData()
        self.axis_select.blockSignals(True)
        self.axis_select.clear()
        for axis in available:
            self.axis_select.addItem(AXIS_LABELS.get(axis, axis), axis)
        if current in available:
            self.axis_select.setCurrentIndex(available.index(current))
        self.axis_select.blockSignals(False)
        return self.axis_select.currentData()

    def combine_mode(self) -> str | None:
        """None (separate curves), "std" or "sem"."""
        return self.combine_select.currentData()

    def plotted_groups(self) -> list[tuple[str, list[Source]]]:
        """Series as plotted: (legend label, loaded sources it covers), combining versions if asked."""
        tables = dict(self.loaded_tables())
        groups = series_groups(self.sources, self.combine_mode() is not None)
        return [(name, kept) for name, members in groups if (kept := [s for s in members if s in tables])]

    def curve_series(self, axis) -> dict[str, list]:
        """Per metric, every plotted series: (label, chart points, color, combined)."""
        tables = dict(self.loaded_tables())
        groups = self.plotted_groups()
        colors = {name: PALETTE[i % len(PALETTE)] for i, (name, _) in enumerate(groups)}
        curves: dict[str, list] = {}
        for metric in self.all_metrics():
            series = []
            for name, members in groups:
                parts = [p for s in members if (p := smooth(tables[s].series(metric, axis), self.smoothing))]
                if not parts:
                    continue
                if len(members) == 1:
                    series.append((name, chart_points(parts[0], metric), colors[name], False))
                else:
                    series.append((name, combined_points(parts, metric, self.combine_mode()), colors[name], True))
            curves[metric] = series
        return curves

    def redraw(self):
        tables = self.loaded_tables()
        axis = self._sync_axes(tables)
        metrics = self.all_metrics()
        visible = [m for m in metrics if m not in self.hidden]
        if not self.sources:
            empty = "Tick one or more runs on the left to plot every metric they logged."
        elif not tables:
            empty = "Loading metrics…"
        elif not metrics:
            empty = "The ticked runs have no numeric metrics."
        else:
            empty = f"All {len(metrics)} plots are hidden. Use Plots ▾ to show them."
        self.grid.set_metrics(visible, empty)
        self.curves = self.curve_series(axis)
        for metric, chart in self.grid.charts().items():
            series = self.curves.get(metric, [])
            chart.band = SPREAD if any(combined for *_, combined in series) else None
            chart.set_metric(metric, chart_title(metric))
            chart.set_series([(name, points, color) for name, points, color, _ in series])
        groups = self.plotted_groups()
        self.legend.setText(
            "   ".join(
                f'<span style="color:{PALETTE[i % len(PALETTE)]}">■</span> {name}' for i, (name, _) in enumerate(groups)
            )
        )
        hidden_note = f" · {len(metrics) - len(visible)} hidden" if len(visible) < len(metrics) else ""
        self.plots_button.setText(f"Plots ({len(visible)}/{len(metrics)}{hidden_note}) ▾" if metrics else "Plots ▾")
        self.update_summary()

    # ── Summary ──────────────────────────────────────────────────────────────

    def summary_entries(self, axis) -> list[tuple[str, str, list]]:
        """(series label, metric, raw points) as plotted, but unsmoothed: combined series use their mean."""
        tables = dict(self.loaded_tables())
        chosen = self.summary.selected_metric()
        entries = []
        for name, members in self.plotted_groups():
            for metric in [chosen] if chosen else self.all_metrics():
                parts = [p for s in members if (p := tables[s].series(metric, axis))]
                if not parts:
                    continue
                points = parts[0] if len(members) == 1 else [(a.x, a.mean) for a in aggregate(parts)]
                entries.append((name, metric, points))
        return entries

    def update_summary(self):
        axis = self.axis_select.currentData()
        self.summary.set_metrics(self.all_metrics())
        entries = self.summary_entries(axis)
        rows = summary_rows(entries, self.summary.threshold_value())
        notes = ["combined series are summarized by their mean curve"] if self.combine_mode() else []
        chosen = self.summary.selected_metric()
        absent = sorted(set([chosen] if chosen else self.all_metrics()) - {metric for _, metric, _ in entries})
        if absent and self.sources:
            # e.g. a value logged once on a row without a step count: say so rather than drop it silently
            notes.append(f"no points on this x-axis for {', '.join(absent)} (pick another x-axis, or see Tables)")
        self.summary.show_rows(rows, axis, " · ".join(notes))

    # ── Export ───────────────────────────────────────────────────────────────

    def curves_csv(self) -> str:
        """Every point of every visible chart, one row per point, as currently drawn."""
        axis = self.axis_select.currentData() or "x"
        mode = self.combine_mode()
        rows = []
        for metric in self.all_metrics():
            if metric in self.hidden:
                continue
            for name, points, _color, combined in self.curves.get(metric, []):
                for p in points:
                    rows.append([name, metric, p["step"], p[metric]] + ([p.get(SPREAD), p.get("n")] if mode else []))
        headers = ["series", "metric", axis, "value"] + ([mode, "runs"] if mode else [])
        return to_csv(headers, rows)

    def summary_csv(self) -> str:
        return to_csv(self.summary.headers, self.summary.rows)

    def export_to(self, kind: str, path: str) -> bool:
        """Write curves or summary CSV, or the charts as PNG, to `path`."""
        try:
            if kind == "png":
                if not self.grid.body.grab().save(path):
                    raise OSError("could not write the image")
            else:
                text = self.curves_csv() if kind == "curves" else self.summary_csv()
                with open(path, "w", encoding="utf-8", newline="") as f:
                    f.write(text)
        except OSError as exc:
            self.status.setText(f"Could not export: {exc}")
            return False
        self.status.setText(f"Exported to {path}")
        return True

    def _export(self, kind: str):
        names = {"curves": "curves.csv", "summary": "summary.csv", "png": "charts.png"}
        filters = "PNG image (*.png)" if kind == "png" else "CSV (*.csv)"
        path, _ = QFileDialog.getSaveFileName(self, "Export", names[kind], filters)
        if path:
            self.export_to(kind, path)

    def _fill_plots_menu(self):
        self.plots_menu.clear()
        metrics = self.all_metrics()
        if not metrics:
            self.plots_menu.addAction("Tick runs to list their plots").setEnabled(False)
            return
        self.plots_menu.addAction("Show all plots", self.show_all)
        self.plots_menu.addSeparator()
        for metric in metrics:
            action = self.plots_menu.addAction(chart_title(metric))
            action.setCheckable(True)
            action.setChecked(metric not in self.hidden)
            action.toggled.connect(lambda on, m=metric: self.set_hidden(m, not on))

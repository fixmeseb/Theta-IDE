"""Spyder-inspired research workspace for deep learning and reinforcement learning experimentation."""
import argparse
import json
import os
import sys
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

if sys.platform == "darwin" and "QTWEBENGINE_CHROMIUM_FLAGS" not in os.environ:
    os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = "--disable-gpu"

from PyQt6.QtCore import QRegularExpression, Qt, QTimer, QUrl
from PyQt6.QtGui import QAction, QDesktopServices, QFont, QIcon, QRegularExpressionValidator
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QStyle,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QToolBar,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from . import wheel_guard
from .about import AboutDialog, AsciiTheta
from .api import DEFAULT_URL, Backend
from .app_icon import DESKTOP_FILE_NAME, ICON_PATH, app_icon, install_desktop_entry, set_windows_app_id
from .components_panel import ComponentsPanel
from .config_tree import ConfigTreeWidget
from .config_viewer import ConfigViewer
from .hub import HubClient, HubDialog
from .model import (
    BASE_EXPERIMENT,
    FINAL_STATUSES,
    LIVE_STATUSES,
    Config,
    Store,
    available_metrics,
    example_runs,
    latest,
    metric_points,
    new_run,
    sample,
)
from .plots import PlotViewer
from .plugins import PluginManager
from .queue_panel import QueuePanel
from .settings import SettingsManager
from .sidetabs import SideTabs, svg_icon
from .tensorboard import TensorBoardPanel
from .terminal import TerminalPanel
from .theme import STYLE, ThemeManager, theme_color
from .theme_builder import ThemeBuilder
from .titlebar import apply_title_bar
from .widgets import Chart, MetricCard, ToggleSlider, YamlHighlighter, label
from .workflows import WorkflowsPanel

# Metrics the monitor's second chart can show: key -> (card title, chart title, subtitle, value format)
SECOND_METRICS = {
    "loss": ("TRAINING LOSS", "Total loss", "policy + 0.5 × value − 0.01 × entropy", ".4f"),
    "value_loss": ("VALUE LOSS", "Value loss", "value-function error · critic MSE loss", ".3f"),
    "policy_loss": ("POLICY LOSS", "Policy loss", "PPO clipped surrogate objective", ".4f"),
    "entropy": ("POLICY ENTROPY", "Policy entropy", "how random actions are · ln 2 ≈ 0.693 is uniform", ".3f"),
    "approx_kl": ("APPROX. KL", "Approximate KL per update", "size of each policy update", ".5f"),
}
# Highest possible evaluation reward per environment, drawn as the reward chart's ceiling.
ENV_MAX_REWARD = {"cartpole": 500}


class Window(QMainWindow):
    def __init__(self, data_dir, api_url=DEFAULT_URL):
        super().__init__()
        self.setWindowTitle("ThetaIDE")
        self.resize(1480, 940)
        self.setMinimumSize(1080, 720)
        self.setDockNestingEnabled(True)
        self.store = Store(data_dir)

        # --- Unified settings (must be first; everything else reads from it) ---
        self.settings_manager = SettingsManager(Path(data_dir))
        self.settings_manager.changed.connect(self._on_settings_changed)

        self.theme_manager = ThemeManager(
            Path(data_dir) / ".appearance.json",
            self,
            initial_theme=self.settings_manager.theme,
        )
        self.pane_sliders = {}
        self.plugin_sliders = {}
        data_p = Path(data_dir).resolve()
        if data_p.name == "runs" and data_p.parent.name == ".thetaide":
            workspace_dir = data_p.parent.parent
            ide_data_dir = data_p.parent
        elif data_p.name == ".thetaide":
            workspace_dir = data_p.parent
            ide_data_dir = data_p
        else:
            workspace_dir = data_p.parent
            ide_data_dir = data_p

        self.plugin_manager = PluginManager(
            self,
            [Path(__file__).parent / "plugins" / "core", Path(__file__).parent / "plugins", ide_data_dir / "plugins"],
            settings_manager=self.settings_manager,
        )
        self.plugin_manager.discover()
        self.hub_client = HubClient(
            workspace_dir=workspace_dir,
            data_dir=ide_data_dir,
            on_change_callback=self._on_hub_component_changed,
            settings_manager=self.settings_manager,
        )
        saved, errors = self.store.load()
        self.runs = saved or example_runs()
        self.active = None
        self.selected = None
        self.pinned_baseline_run = None
        self.timer = QTimer(self)
        self.timer.setInterval(350)
        self.timer.timeout.connect(self.tick)
        self.backend = Backend(api_url, self)
        self.schema = None
        self.schema_pending = False
        self.compose_serial = 0
        self.compose_timer = QTimer(self)
        self.compose_timer.setSingleShot(True)
        self.compose_timer.setInterval(250)
        self.compose_timer.timeout.connect(self.request_compose)
        self.compose_valid = False
        self.poll_timer = QTimer(self)
        self.poll_timer.setInterval(1000)
        self.poll_timer.timeout.connect(self.poll_job)
        self.poll_busy = False
        self.queue_timer = QTimer(self)
        self.queue_timer.setInterval(2000)
        self.queue_timer.timeout.connect(self.poll_queue)
        self.queue_busy = False
        self.issued_ids = set()
        self.queue_running = False
        self.active_config_path = None
        self.active_config_rel_path = None
        self.current_experiment = None
        self.fixed_fields = {}
        self.fields = {}
        self.make_center()
        self.plugin_manager.initialize_plugins()
        self.init_default_experiment()
        self.tabs.tabOrderChanged.connect(lambda _: self.save_layout())
        self.load_layout()
        if hasattr(self, "settings_manager"):
            self.tabs.set_auto_hide(bool(self.settings_manager.get("sidebar", "auto_hide", default=False)))
        self.tabs.autoHideChanged.connect(self.on_sidebar_auto_hide_changed)
        self.make_menus()
        self.theme_status = label("", "muted")
        self.statusBar().addWidget(self.theme_status)
        self.theme_manager.changed.connect(self.theme_changed)
        self.theme_manager.committed.connect(self.persist_theme_choice)
        self.theme_changed()
        self.state_label = label("TRAINING   •   idle  ", "muted")
        self.statusBar().addPermanentWidget(self.state_label)
        self.backend_label = label("BACKEND   ○   connecting…  ", "muted")
        self.statusBar().addPermanentWidget(self.backend_label)
        self.refresh_runs()
        self.update_config()
        if self.runs:
            self.select_run(self.runs[0])
        self.log("ThetaIDE ready. Launch trains on this machine through the backend; "
                 "Run → Start simulated demo needs no backend.")
        self.request_schema(apply_defaults=True)
        self.reconnect_live_run()
        for error in errors:
            self.log(error)
        if self.theme_manager.error:
            self.log(self.theme_manager.error)
        self.default_layout = self.saveState()

        # --- Hotkey and Action-key Navigation Manager ---
        from .hotkeys import HotkeyManager
        self.hotkey_manager = HotkeyManager(self, self.settings_manager)

    def button(self, text, callback, primary=False):
        button = QPushButton(text)
        if primary:
            button.setObjectName("primary")
        button.clicked.connect(callback)
        return button

    def make_center(self):
        self.tabs = SideTabs()
        self.tabs.logoClicked.connect(self.show_about)
        self.setCentralWidget(self.tabs)

        # 1. Components Panel
        self.components_panel = ComponentsPanel(hub_client=self.hub_client, log_fn=self.log, parent=self)
        self.tabs.addTab(self.components_panel, "Components", "components", "Components", tab_id="components")

        # 2. Experiment (Config) Panel
        config_panel = QWidget()
        config_layout = QVBoxLayout(config_panel)
        config_layout.setContentsMargins(18, 14, 18, 14)
        config_layout.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.config_splitter = splitter

        # 1. Config Tree (mirrors in/config/experiment/) - extends to top!
        self.config_tree = ConfigTreeWidget(mode="experiments")
        self.config_tree.setMinimumWidth(220)
        self.tree = self.config_tree.tree  # backwards compatibility alias
        self.config_tree.file_selected.connect(self.on_config_file_selected)
        splitter.addWidget(self.config_tree)

        # 2. Right Side Section (contains actions bar and viewer / preview)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)

        actions_bar = QHBoxLayout()
        actions_bar.setSpacing(8)
        self.start_button = self.button("Launch training", self.launch_training, True)
        self.start_button.setToolTip("Train the loaded experiment config through the backend (F5)")
        self.start_button.setEnabled(False)
        actions_bar.addWidget(self.start_button)
        self.queue_button = self.button("Add to queue", self.add_to_queue)
        self.queue_button.setToolTip("Queue the loaded config; queued jobs train one at a time, in order (Ctrl+Shift+Q)")
        self.queue_button.setEnabled(False)
        actions_bar.addWidget(self.queue_button)
        self.stop_button = self.button("Stop", self.stop_run)
        self.stop_button.setEnabled(False)
        actions_bar.addWidget(self.stop_button)
        actions_bar.addWidget(self.button("Export recipe YAML…", self.export_config))
        actions_bar.addStretch()

        self.btn_toggle_yaml = QPushButton("View Hydra YAML")
        self.btn_toggle_yaml.setCheckable(True)
        self.btn_toggle_yaml.setChecked(False)
        self.btn_toggle_yaml.setToolTip("Toggle preview of the resolved Hydra YAML configuration")
        self.btn_toggle_yaml.clicked.connect(lambda: self.toggle_yaml_preview())
        actions_bar.addWidget(self.btn_toggle_yaml)

        right_layout.addLayout(actions_bar)

        preview_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.preview_splitter = preview_splitter

        # Boxed Config Viewer (takes up the main area of the screen)
        self.config_viewer = ConfigViewer()
        self.config_viewer.config_changed.connect(self.update_config)
        self.config_viewer.save_requested.connect(self.on_config_saved)
        preview_splitter.addWidget(self.config_viewer)

        # Preview Panel (Hydra YAML - hidden by default!)
        preview_panel = QWidget()
        preview_layout = QVBoxLayout(preview_panel)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.setSpacing(6)

        prev_header = QHBoxLayout()
        prev_header.setSpacing(8)
        prev_header.addWidget(label("RESOLVED HYDRA YAML", "eyebrow"))
        prev_header.addStretch()
        btn_close_prev = QToolButton()
        btn_close_prev.setText("✕")
        btn_close_prev.setToolTip("Close preview and expand config viewer")
        btn_close_prev.clicked.connect(lambda: self.toggle_yaml_preview(False))
        prev_header.addWidget(btn_close_prev)
        preview_layout.addLayout(prev_header)

        self.preview_status = label("Resolved config • waiting for backend", "muted")
        preview_layout.addWidget(self.preview_status)
        self.preview_errors = label("", "configError")
        self.preview_errors.setWordWrap(True)
        self.preview_errors.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.preview_errors.hide()
        preview_layout.addWidget(self.preview_errors)
        self.preview = QPlainTextEdit()
        self.preview.setReadOnly(True)
        self.highlighter = YamlHighlighter(self.preview.document())
        preview_layout.addWidget(self.preview, 1)

        self.builder_status = label("", "muted")
        self.builder_status.setWordWrap(True)
        preview_layout.addWidget(self.builder_status)

        self.preview_panel = preview_panel
        self.preview_panel.hide()  # Hidden by default!
        preview_splitter.addWidget(preview_panel)

        preview_splitter.setSizes([1000, 0])
        right_layout.addWidget(preview_splitter, 1)

        splitter.addWidget(right_panel)
        splitter.setSizes([260, 1000])
        config_layout.addWidget(splitter, 1)
        self.config_panel = config_panel
        self.tabs.addTab(self.config_panel, "Experiment", "config", "Experiment", tab_id="config")

        # 3. Workflows Panel (String Diagrams)
        self.workflows_panel = WorkflowsPanel(log_fn=self.log, parent=self)
        self.tabs.addTab(self.workflows_panel, "Workflows", "workflow", "Workflows", tab_id="workflows")

        # 4. Training Monitor Panel
        monitor = QWidget()
        layout = QVBoxLayout(monitor)
        layout.setContentsMargins(20, 18, 20, 14)
        layout.setSpacing(14)
        # Eyebrow row with curve smoothing slider
        eyebrow_row = QHBoxLayout()
        eyebrow_row.addWidget(label("EXPERIMENT / OVERVIEW", "eyebrow"))
        eyebrow_row.addStretch()
        self.smoothing_label = label("Smoothing: 0%", "muted")
        eyebrow_row.addWidget(self.smoothing_label)
        self.smoothing_slider = QSlider(Qt.Orientation.Horizontal)
        self.smoothing_slider.setRange(0, 95)
        self.smoothing_slider.setValue(0)
        self.smoothing_slider.setFixedWidth(110)
        self.smoothing_slider.setToolTip("Exponential moving average smoothing for training curves (0% = raw)")
        self.smoothing_slider.valueChanged.connect(self.on_smoothing_changed)
        eyebrow_row.addWidget(self.smoothing_slider)
        layout.addLayout(eyebrow_row)

        # Run navigator toolbar: Run selector combo, prev/next, jump to live, pin baseline
        nav_row = QHBoxLayout()
        nav_row.setSpacing(8)
        self.run_combo = QComboBox()
        self.run_combo.setMinimumWidth(320)
        self.run_combo.setToolTip("Select any current or past experiment run to inspect")
        self.run_combo.currentIndexChanged.connect(self.on_run_combo_changed)
        nav_row.addWidget(self.run_combo, 1)

        self.btn_prev_run = QPushButton("Previous")
        self.btn_prev_run.setToolTip("View previous experiment run in history")
        self.btn_prev_run.clicked.connect(self.select_prev_run)
        nav_row.addWidget(self.btn_prev_run)

        self.btn_next_run = QPushButton("Next")
        self.btn_next_run.setToolTip("View next experiment run in history")
        self.btn_next_run.clicked.connect(self.select_next_run)
        nav_row.addWidget(self.btn_next_run)

        self.btn_live_jump = QPushButton("Jump to live")
        self.btn_live_jump.setToolTip("Return view to the currently training run")
        self.btn_live_jump.clicked.connect(self.jump_to_live_run)
        self.btn_live_jump.hide()
        nav_row.addWidget(self.btn_live_jump)

        self.btn_pin_baseline = QPushButton("Pin as baseline")
        self.btn_pin_baseline.setCheckable(True)
        self.btn_pin_baseline.setToolTip("Pin this run to overlay as a dashed baseline curve on other runs")
        self.btn_pin_baseline.clicked.connect(self.toggle_pin_baseline)
        nav_row.addWidget(self.btn_pin_baseline)

        layout.addLayout(nav_row)

        self.run_title = label("Your next experiment", "heading")
        layout.addWidget(self.run_title)

        caption_row = QHBoxLayout()
        self.run_caption = label("Configure an experiment, then launch training.", "muted")
        caption_row.addWidget(self.run_caption, 1)
        self.storage_label = label("results/logs/  ·  results/jobs/jobs.db", "muted")
        caption_row.addWidget(self.storage_label)
        layout.addLayout(caption_row)
        row = QHBoxLayout()
        self.reward_card = MetricCard("EPISODE REWARD", "synthetic evaluation / mean")
        self.loss_card = MetricCard("TRAINING LOSS", "total loss across minibatches")
        self.steps_card = MetricCard("TIMESTEPS", "configured training budget")
        for card in (self.reward_card, self.loss_card, self.steps_card):
            row.addWidget(card)
        layout.addLayout(row)
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(5)
        layout.addWidget(self.progress)
        self.reward_chart = Chart("reward", "Episode reward  ·  mean ± 1 std over evaluation episodes",
                                  band="reward_std")
        self.second_metric = "loss"
        self.loss_chart = Chart(self.second_metric, SECOND_METRICS[self.second_metric][1], zero_based=False)
        self.metric_select = QComboBox()
        self.metric_select.setToolTip("Metric shown in this chart and the second card")
        self.metric_select.currentIndexChanged.connect(self.second_metric_changed)
        self.loss_chart.set_corner_widget(self.metric_select)
        layout.addWidget(self.reward_chart, 3)
        layout.addWidget(self.loss_chart, 2)
        self.monitor_note = label("", "muted")
        self.monitor_note.setWordWrap(True)
        monitor_footer = QHBoxLayout()
        monitor_footer.addWidget(self.monitor_note, 1)
        monitor_footer.addWidget(self.button("Open TensorBoard", self.show_tensorboard))
        layout.addLayout(monitor_footer)
        self.monitor_panel = monitor
        self.tabs.addTab(self.monitor_panel, "Training monitor", "monitor", "Monitor", tab_id="monitor")

        # 4. Results Browser Panel
        results = QWidget()
        results_layout = QVBoxLayout(results)
        results_layout.setContentsMargins(18, 18, 18, 18)
        results_layout.addWidget(label("Experiment history", "heading"))
        results_layout.addWidget(label("Select one run to inspect, or two to compare (Ctrl + click).", "muted"))
        self.search = QLineEdit()
        self.search.setPlaceholderText("Filter by experiment name, seed, or status…")
        self.search.textChanged.connect(self.refresh_runs)
        results_layout.addWidget(self.search)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Experiment", "Seed", "Status", "Reward", "Source"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.table.verticalHeader().hide()
        self.table.verticalHeader().setDefaultSectionSize(40)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.setShowGrid(False)
        self.table.itemSelectionChanged.connect(self.table_selected)
        self.table.itemDoubleClicked.connect(lambda _: self.tabs.setCurrentWidget(self.monitor_panel))
        results_layout.addWidget(self.table, 1)
        actions = QHBoxLayout()
        actions.addWidget(self.button("Load saved config", self.load_selected_config))
        actions.addWidget(self.button("Compare two runs", self.compare))
        actions.addWidget(self.button("View plot", self.view_selected_plot))
        actions.addStretch()
        results_layout.addLayout(actions)
        self.results_panel = results
        self.tabs.addTab(self.results_panel, "Results browser", "results", "Results", tab_id="results")

        # 5. Plot Viewer
        self.plot_viewer = PlotViewer()
        self.tabs.addTab(self.plot_viewer, "Plot viewer", "plots", "Plots", tab_id="plots")

        # 6. TensorBoard Panel
        self.tensorboard_panel = TensorBoardPanel(self.backend, self.log)
        self.tabs.addTab(self.tensorboard_panel, "TensorBoard", "tensorboard", "TensorBoard", tab_id="tensorboard")

        # 7. Job Queue Panel
        self.queue_panel = QueuePanel(self.move_queued, self.remove_queued, self.open_job, self.set_queue_running)
        self.tabs.addTab(self.queue_panel, "Job queue", "queue", "Queue", tab_id="queue")

        # 8. Terminal Panel
        self.terminal_panel = TerminalPanel(cwd=str(Path.cwd()), parent=self)
        self.tabs.addTab(self.terminal_panel, "Terminal", "terminal", "Terminal", tab_id="terminal")

        # 9. Console Panel
        self.console_panel = self.make_console()
        self.tabs.addTab(self.console_panel, "Console", "console", "Console", tab_id="console")

        self.settings_panel = self.make_settings()
        self.tabs.set_settings_widget(self.settings_panel)
        self.tabs.logoClicked.connect(lambda: getattr(self, "settings_view", None) and self.settings_view.reset_to_ascii())
        self.tabs.currentChanged.connect(self.tab_changed)

    def dock(self, title, name, widget, area):
        dock = QDockWidget(title, self)
        dock.setObjectName(name)
        dock.setWidget(widget)
        self.addDockWidget(area, dock)
        return dock

    def init_default_experiment(self):
        default_exp = "experiment/cartpole/quick_test.yaml"
        if not (self.config_tree.root_dir / default_exp).exists():
            default_exp = "experiment/thetaide/cartpole_ppo_reference.yaml"
        self.config_tree.select_file(default_exp)

    def toggle_yaml_preview(self, checked=None):
        if checked is None:
            checked = self.btn_toggle_yaml.isChecked()
        else:
            self.btn_toggle_yaml.setChecked(checked)
        self.preview_panel.setVisible(checked)
        if hasattr(self, "preview_splitter"):
            if checked:
                self.preview_splitter.setSizes([600, 420])
                self.request_compose()
            else:
                self.preview_splitter.setSizes([1000, 0])

    def on_config_file_selected(self, file_path, rel_path):
        self.active_config_path = file_path
        self.active_config_rel_path = rel_path
        self.config_viewer.load_file(file_path, rel_path)

        norm_rel = Path(rel_path).as_posix()
        if norm_rel.startswith("experiment/") and not Path(rel_path).name.startswith("_"):
            exp_name = Path(norm_rel).relative_to("experiment").with_suffix("").as_posix()
            self.current_experiment = exp_name
            self.start_button.setEnabled(self.compose_valid)
            self.queue_button.setEnabled(self.compose_valid)
            self.start_button.setToolTip(f"Train {exp_name} through the backend (F5)")
        else:
            self.current_experiment = None
            self.start_button.setEnabled(False)
            self.queue_button.setEnabled(False)
            self.start_button.setToolTip("Select an experiment from experiment/ to launch training")

        self.request_compose()
        if hasattr(self, "plugin_manager"):
            self.plugin_manager.notify_experiment_changed(self.current_experiment, self.active_config_path)

    def duplicate_experiment(self):
        self.config_tree.prompt_duplicate()

    def save_current_config(self):
        if hasattr(self, "config_viewer"):
            ok = self.config_viewer.save_to_disk()
            if ok:
                self.statusBar().showMessage(f"Saved {self.config_viewer.current_rel_path}", 4000)

    def on_config_saved(self):
        self.statusBar().showMessage(f"Saved {self.config_viewer.current_rel_path}", 4000)
        self.request_compose()

    def make_console(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(18, 16, 18, 14)
        layout.setSpacing(10)

        header = QHBoxLayout()
        header.setSpacing(8)
        header.addWidget(label("CONSOLE / SYSTEM OUTPUT", "eyebrow"))
        header.addStretch()
        for cmd_name, desc in (
            ("status", "Show active training and run summary"),
            ("config", "Show pipeline config command"),
            ("help", "List available console commands"),
        ):
            btn = QPushButton(f"› {cmd_name}")
            btn.setToolTip(desc)
            btn.clicked.connect(lambda _, c=cmd_name: self.run_quick_command(c))
            header.addWidget(btn)
        header.addWidget(self.button("Clear output", self.console_clear))
        layout.addLayout(header)

        self.console = QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.setMaximumBlockCount(3000)
        layout.addWidget(self.console, 1)

        self.command = QLineEdit()
        self.command.setPlaceholderText("Console › help, status, config, clear (see Terminal tab for system shell)")
        self.command.returnPressed.connect(self.console_command)
        layout.addWidget(self.command)
        return panel

    def console_clear(self):
        if hasattr(self, "console"):
            self.console.clear()

    def make_settings(self):
        from .settings_view import SettingsView
        self.settings_view = SettingsView(self)
        return self.settings_view


    def settings_theme_selected(self, theme_name):
        if theme_name and theme_name != self.theme_manager.active["name"]:
            theme = self.theme_manager.themes().get(theme_name)
            if theme:
                self.select_theme(theme)

    def _on_settings_changed(self):
        """Called when settings.toml changes on disk (or when the app mutates it).

        Applies any differences to the live UI — theme, sidebar visibility, etc.
        Designed to be idempotent so redundant calls are harmless.
        """
        if not hasattr(self, "settings_manager"):
            return

        # --- Theme hot-reload ---
        new_theme = self.settings_manager.theme
        if new_theme != self.theme_manager.active.get("name"):
            if not self.theme_manager.select_by_name(new_theme):
                self.log(f"[settings] Unknown theme '{new_theme}' in settings.toml; keeping current theme.")
            else:
                # Sync the dropdown in the Settings panel if it's already built
                if hasattr(self, "settings_theme_select"):
                    self.settings_theme_select.blockSignals(True)
                    self.settings_theme_select.setCurrentText(new_theme)
                    self.settings_theme_select.blockSignals(False)

        # --- Sidebar visibility hot-reload ---
        known_set = set(self.settings_manager.sidebar_order)
        visible_set = set(self.settings_manager.sidebar_visible)
        if known_set and hasattr(self, "tabs"):
            for pid in self.tabs.tab_order:
                if pid not in known_set:
                    continue  # plugin tab not in settings — leave it alone
                should_be = pid in visible_set
                if self.tabs.is_tab_visible(pid) != should_be:
                    self.tabs.set_tab_visible(pid, should_be)
                    if hasattr(self, "pane_sliders") and pid in self.pane_sliders:
                        slider = self.pane_sliders[pid]
                        slider.blockSignals(True)
                        slider.setChecked(should_be)
                        slider.blockSignals(False)
            if hasattr(self, "pane_sliders"):
                self.update_pane_sliders_state()

        # --- Hotkeys hot-reload ---
        if hasattr(self, "settings_action_key_combo"):
            cur = self.settings_manager.action_key.lower()
            idx = self.settings_action_key_combo.findData(cur)
            if idx >= 0:
                if self.settings_action_key_combo.currentIndex() != idx:
                    self.settings_action_key_combo.blockSignals(True)
                    self.settings_action_key_combo.setCurrentIndex(idx)
                    self.settings_action_key_combo.blockSignals(False)
            else:
                self.settings_action_key_combo.blockSignals(True)
                self.settings_action_key_combo.addItem(cur.replace("_", "+").upper(), cur)
                self.settings_action_key_combo.setCurrentIndex(self.settings_action_key_combo.count() - 1)
                self.settings_action_key_combo.blockSignals(False)

        # --- Menu Shortcuts hot-reload ---
        if hasattr(self, "menu_actions"):
            for aid, action in self.menu_actions.items():
                sc = self.settings_manager.get("shortcuts", aid)
                if sc is not None:
                    action.setShortcut(sc)

        # --- Settings view sync ---
        if hasattr(self, "settings_view") and hasattr(self.settings_view, "sync_from_settings"):
            self.settings_view.sync_from_settings()

    def _on_action_key_changed(self, index):
        if not hasattr(self, "settings_action_key_combo"):
            return
        val = self.settings_action_key_combo.itemData(index)
        if val:
            self.settings_manager.set("hotkeys", "action_key", val)
            self.statusBar().showMessage(f"Action Key set to {val.replace('_', ' ').title()}", 3000)

    def switch_to_pane(self, target: str | int) -> bool:
        """Switch to a pane by name or numeric index [0, 1, 2, ...]."""
        if hasattr(self, "hotkey_manager"):
            if isinstance(target, int):
                return self.hotkey_manager.switch_to_pane_by_index(target)
            return self.hotkey_manager.switch_to_pane(str(target))
        return False

    def on_sidebar_auto_hide_changed(self, enabled):
        """Keep the menu item and the Settings switch in step, and remember the choice."""
        for control in (getattr(self, "auto_hide_action", None), getattr(self, "auto_hide_slider", None)):
            if control is not None and control.isChecked() != enabled:
                control.blockSignals(True)
                control.setChecked(enabled)
                control.blockSignals(False)
        if hasattr(self, "settings_manager"):
            self.settings_manager.set("sidebar", "auto_hide", enabled)
        self.statusBar().showMessage(
            "Sidebar auto-hide on: hover the left edge to show it." if enabled else "Sidebar docked.", 4000)

    def on_pane_slider_toggled(self, pane_id, checked):
        # Count the other panes, so the guard holds whether or not this slider has already flipped
        others_visible = sum(1 for pid, s in self.pane_sliders.items() if pid != pane_id and s.isChecked())
        if not checked and others_visible == 0:
            slider = self.pane_sliders.get(pane_id)
            if slider:
                slider.blockSignals(True)
                slider.setChecked(True)
                slider.blockSignals(False)
            self.statusBar().showMessage("At least one panel must remain visible in the sidebar.", 3000)
            return

        self.tabs.set_tab_visible(pane_id, checked)
        self.update_pane_sliders_state()
        self.save_layout()

    def on_plugin_slider_toggled(self, plugin_id, checked):
        if not hasattr(self, "plugin_manager"):
            return
        if checked:
            success = self.plugin_manager.enable_plugin(plugin_id)
            if success:
                name = self.plugin_manager.manifests[plugin_id].name
                self.statusBar().showMessage(f"Activated plugin: {name}", 4000)
            else:
                slider = self.plugin_sliders.get(plugin_id)
                if slider:
                    slider.blockSignals(True)
                    slider.setChecked(False)
                    slider.blockSignals(False)
                self.statusBar().showMessage(f"Failed to activate plugin: {plugin_id}", 4000)
        else:
            self.plugin_manager.disable_plugin(plugin_id)
            manifest = self.plugin_manager.manifests.get(plugin_id)
            name = manifest.name if manifest else plugin_id
            self.statusBar().showMessage(f"Deactivated plugin: {name}", 4000)
        self.save_layout()

    def _clear_layout(self, layout):
        if not layout:
            return
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                self._clear_layout(item.layout())

    def _create_plugin_row(self, pid: str, manifest, is_core: bool = False):
        row = QHBoxLayout()
        row.setSpacing(10)

        info_layout = QVBoxLayout()
        info_layout.setSpacing(1)
        badge = "  ·  Core" if is_core else ""
        title_lbl = label(f"{manifest.name}  ·  v{manifest.version}{badge}")
        title_lbl.setStyleSheet("font-weight: 500;")
        info_layout.addWidget(title_lbl)
        if manifest.description:
            desc_lbl = label(manifest.description, "muted")
            info_layout.addWidget(desc_lbl)
        row.addLayout(info_layout, 1)

        is_enabled = self.plugin_manager.is_plugin_enabled(pid)
        slider = ToggleSlider(checked=is_enabled)
        slider.setToolTip(f"Enable or disable {manifest.name}")
        slider.setAccessibleName(f"Toggle {manifest.name} plugin")
        slider.toggled.connect(lambda chk, p=pid: self.on_plugin_slider_toggled(p, chk))
        self.plugin_sliders[pid] = slider
        row.addWidget(slider)

        instance = self.plugin_manager.instances.get(pid)
        has_settings = False
        if instance and hasattr(instance, "get_settings_widget"):
            has_settings = True
        elif hasattr(manifest, "extra") and isinstance(manifest.extra, dict) and manifest.extra.get("has_settings", False):
            has_settings = True

        if has_settings:
            btn_settings = QToolButton()
            btn_settings.setIcon(svg_icon("settings", {(QIcon.Mode.Normal, QIcon.State.Off): "text"}))
            btn_settings.setToolTip(f"{manifest.name} Settings")
            btn_settings.setFixedSize(30, 30)
            btn_settings.setCursor(Qt.CursorShape.PointingHandCursor)
            btn_settings.clicked.connect(lambda _, p=pid: self.open_plugin_settings(p))
            row.addWidget(btn_settings)

        if not is_core:
            btn_trash = QToolButton()
            trash_icon_path = Path(__file__).parent / "icons" / "trash.svg"
            if trash_icon_path.exists():
                btn_trash.setIcon(QIcon(str(trash_icon_path)))
            else:
                btn_trash.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_TrashIcon))
            btn_trash.setToolTip(f"Uninstall {manifest.name}")
            btn_trash.setFixedSize(28, 28)
            btn_trash.setCursor(Qt.CursorShape.PointingHandCursor)
            btn_trash.setStyleSheet(
                "QToolButton { border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 4px; background: rgba(255, 255, 255, 0.04); } "
                "QToolButton:hover { background: rgba(251, 73, 52, 0.2); border-color: #fb4934; }"
            )
            btn_trash.clicked.connect(lambda _, p=pid: self.uninstall_plugin_from_settings(p))
            row.addWidget(btn_trash)

        return row

    def refresh_plugins_ui(self):
        self.plugin_sliders = {}

        has_core = hasattr(self, "core_plugins_grid") and self.core_plugins_grid is not None
        has_community = hasattr(self, "plugins_grid") and self.plugins_grid is not None

        if not has_core and not has_community:
            return

        if has_core:
            self._clear_layout(self.core_plugins_grid)
        if has_community:
            self._clear_layout(self.plugins_grid)

        core_manifests = {}
        community_manifests = {}

        if hasattr(self, "plugin_manager") and self.plugin_manager.manifests:
            for pid, manifest in self.plugin_manager.manifests.items():
                kind = getattr(manifest, "kind", None)
                if not kind and hasattr(manifest, "extra") and isinstance(manifest.extra, dict):
                    kind = manifest.extra.get("kind", "plugin")
                if kind is not None and kind != "plugin":
                    continue

                if getattr(manifest, "is_core", False):
                    core_manifests[pid] = manifest
                else:
                    community_manifests[pid] = manifest

        # 1. Render Core Plugins
        if has_core:
            if core_manifests:
                for pid, manifest in core_manifests.items():
                    row = self._create_plugin_row(pid, manifest, is_core=True)
                    self.core_plugins_grid.addLayout(row)
            else:
                self.core_plugins_grid.addWidget(label("No core plugins installed.", "muted"))

        # 2. Render Community Plugins
        if has_community:
            if community_manifests:
                for pid, manifest in community_manifests.items():
                    row = self._create_plugin_row(pid, manifest, is_core=False)
                    self.plugins_grid.addLayout(row)
            else:
                self.plugins_grid.addWidget(label("No community plugins installed.", "muted"))

    def uninstall_plugin_from_settings(self, plugin_id: str):
        if not hasattr(self, "plugin_manager"):
            return
        manifest = self.plugin_manager.manifests.get(plugin_id)
        name = manifest.name if manifest else plugin_id
        reply = QMessageBox.question(
            self,
            "Uninstall Plugin",
            f"Are you sure you want to uninstall '{name}'?\nThis will remove its files and disable the plugin.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self.plugin_manager.uninstall_plugin(plugin_id)

        # Let the Hub refresh its installed-status badge if it's open.
        if hasattr(self, "hub_client"):
            self.hub_client.refresh_installed_status()

        self.refresh_plugins_ui()
        self.save_layout()
        self.statusBar().showMessage(f"Uninstalled plugin: {name}", 4000)

    def open_plugin_settings(self, plugin_id: str):
        from PyQt6.QtWidgets import QDialog, QDialogButtonBox, QMessageBox, QVBoxLayout
        
        instance = self.plugin_manager.instances.get(plugin_id)
        if not instance:
            QMessageBox.information(self, "Plugin Disabled", "Please enable the plugin first to configure its settings.")
            return
            
        context = self.plugin_manager.contexts.get(plugin_id)
        settings_widget = None
        if hasattr(instance, "get_settings_widget"):
            settings_widget = instance.get_settings_widget(context)
            
        if not settings_widget:
            QMessageBox.information(self, "No Settings", f"The plugin '{plugin_id}' has no configurable settings.")
            return
            
        dialog = QDialog(self)
        manifest = self.plugin_manager.manifests.get(plugin_id)
        name = manifest.name if manifest else plugin_id
        dialog.setWindowTitle(f"{name} Settings")
        dialog.setMinimumWidth(400)
        
        layout = QVBoxLayout(dialog)
        layout.addWidget(settings_widget)
        
        btn_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        btn_box.accepted.connect(dialog.accept)
        layout.addWidget(btn_box)
        
        dialog.exec()

    def _on_hub_component_changed(self, component_id: str, action: str):
        comp = self.hub_client.get_component(component_id) if hasattr(self, "hub_client") else None

        # Determine target category folder in in/config/ (e.g. "agent", "model", "env", "experiment")
        target_folder = None
        target_config_rel = None
        if comp:
            folder_map = {
                "method": "agent",
                "model": "model",
                "env": "env",
                "experiment": "experiment",
            }
            target_folder = folder_map.get(comp.kind)
            if target_folder:
                target_config_rel = f"{target_folder}/{comp.id}.yaml"

        # Refresh components panel and config tree so added or removed YAMLs update immediately without collapsing folders
        if hasattr(self, "components_panel") and self.components_panel is not None:
            self.components_panel.reload_components(
                target_rel_path=target_config_rel if action == "install" else None,
                ensure_expanded=target_folder,
            )
        if hasattr(self, "config_tree") and self.config_tree is not None:
            self.config_tree.populate(ensure_expanded=target_folder)

        # Non-plugin components (methods, models, envs) are filesystem assets only — no plugin UI to update.
        if comp and comp.kind != "plugin":
            self.statusBar().showMessage(f"Hub {action} finished: {component_id}", 4000)
            return

        if not hasattr(self, "plugin_manager"):
            return

        if action == "uninstall":
            # Resolve the plugin id from the component's target_path (e.g. "plugins/my_plugin" → "my_plugin").
            # Deactivate any running instance before discover() prunes its manifest.
            pid = Path(comp.target_path).name if comp and comp.target_path else component_id
            if pid in self.plugin_manager.instances:
                self.plugin_manager.disable_plugin(pid)
            self.plugin_manager.manifests.pop(pid, None)
            self.plugin_manager.enabled_states.pop(pid, None)
            self.plugin_manager.save_states()

        # Sync manifests with the current filesystem state.
        self.plugin_manager.discover()

        if action == "install":
            # Auto-enable if the user had it enabled before (e.g. re-installing).
            pid = Path(comp.target_path).name if comp and comp.target_path else component_id
            if pid in self.plugin_manager.manifests:
                if self.plugin_manager.is_plugin_enabled(pid) and pid not in self.plugin_manager.instances:
                    self.plugin_manager.enable_plugin(pid)

        self.refresh_plugins_ui()
        self.save_layout()
        self.statusBar().showMessage(f"Hub {action} finished: {component_id}", 4000)

    def open_hub(self, initial_kind: str | None = None):
        if hasattr(self, "hub_client"):
            dialog = HubDialog(self.hub_client, initial_kind=initial_kind, parent=self)
            dialog.exec()

    def update_pane_sliders_state(self):
        visible_sliders = [s for s in self.pane_sliders.values() if s.isChecked()]
        if len(visible_sliders) == 1:
            visible_sliders[0].setEnabled(False)
            visible_sliders[0].setToolTip("At least one panel must remain visible in the sidebar")
        else:
            for s in self.pane_sliders.values():
                s.setEnabled(True)
                s.setToolTip("Show or hide this panel in the sidebar")

    def reset_sidebar_layout(self):
        default_order = [
            "components",
            "config",
            "workflows",
            "monitor",
            "results",
            "plots",
            "tensorboard",
            "queue",
            "terminal",
            "console",
        ]
        self.tabs.apply_tab_order(default_order)
        for pid, slider in self.pane_sliders.items():
            slider.blockSignals(True)
            slider.setChecked(True)
            slider.setEnabled(True)
            slider.blockSignals(False)
            self.tabs.set_tab_visible(pid, True)
        self.save_layout()
        self.statusBar().showMessage("Restored default sidebar panels and order.", 4000)

    def save_layout(self):
        if not hasattr(self, "settings_manager"):
            return
        try:
            order = list(self.tabs.tab_order)
            visible = [pid for pid in order if self.tabs.is_tab_visible(pid)]
            self.settings_manager.set("sidebar", "order", order)
            self.settings_manager.set("sidebar", "visible", visible)
        except Exception as exc:
            self.log(f"Failed to save sidebar layout: {exc}")

    def load_layout(self):
        if not hasattr(self, "settings_manager"):
            return
        try:
            order = self.settings_manager.sidebar_order
            visible_set = set(self.settings_manager.sidebar_visible)
            known_set = set(order)  # tabs settings.toml explicitly knows about
            if order:
                self.tabs.apply_tab_order(order)
            for pid in self.tabs.tab_order:
                if pid not in known_set:
                    # Plugin-added tab not yet in settings — leave it visible
                    # (the user hasn't made a preference yet; save_layout will
                    # persist it the next time the layout changes)
                    continue
                is_visible = pid in visible_set
                self.tabs.set_tab_visible(pid, is_visible)
                if hasattr(self, "pane_sliders") and pid in self.pane_sliders:
                    slider = self.pane_sliders[pid]
                    slider.blockSignals(True)
                    slider.setChecked(is_visible)
                    slider.blockSignals(False)
            if hasattr(self, "pane_sliders"):
                self.update_pane_sliders_state()
        except Exception as exc:
            self.log(f"Failed to load sidebar layout: {exc}")

    def test_backend_connection(self):
        if hasattr(self, "settings_backend_status"):
            self.settings_backend_status.setText("Status: Testing connection…")
            self.settings_backend_status.setStyleSheet("")
            self.backend.get("/api/health", self.backend_health_callback)

    def backend_health_callback(self, data, error):
        if not hasattr(self, "settings_backend_status"):
            return
        if error:
            self.settings_backend_status.setText(f"Status: Disconnected ({error})")
            self.settings_backend_status.setStyleSheet("color: #fb4934;")
        else:
            status_text = data.get("status", "ok") if isinstance(data, dict) else "ok"
            self.settings_backend_status.setText(f"Status: Connected (Server status: {status_text})")
            self.settings_backend_status.setStyleSheet(f"color: {theme_color('primary')};")

    def open_swagger_docs(self):
        docs_url = f"{self.backend.base_url.rstrip('/')}/docs"
        QDesktopServices.openUrl(QUrl(docs_url))

    def run_quick_command(self, command):
        self.tabs.setCurrentWidget(self.console_panel)
        if hasattr(self, "command"):
            self.command.setText(command)
        self.console_command(command)

    def toggle_ascii_animation(self, paused):
        if hasattr(self, "settings_ascii"):
            self.settings_ascii.set_paused(paused)
        if hasattr(self, "anim_toggle_btn") and hasattr(self.anim_toggle_btn, "isCheckable") and self.anim_toggle_btn.isCheckable():
            self.anim_toggle_btn.setText("Resume animation" if paused else "Pause animation")

    def make_menus(self):
        self.menu_actions = {}
        file_menu = self.menuBar().addMenu("File")
        for aid, title, default_sc, callback in (
            ("new_experiment", "New experiment", "Ctrl+N", self.new_experiment),
            ("export_config", "Export draft YAML…", "Ctrl+Shift+S", self.export_config),
            ("save_config", "Save configuration", "Ctrl+S", self.save_current_config),
            ("quit", "Quit", "Ctrl+Q", self.close),
        ):
            action = QAction(title, self)
            sc = self.settings_manager.get("shortcuts", aid, default=default_sc) if hasattr(self, "settings_manager") else default_sc
            if sc:
                action.setShortcut(sc)
            action.triggered.connect(callback)
            file_menu.addAction(action)
            self.menu_actions[aid] = action

        run_menu = self.menuBar().addMenu("Run")
        for aid, title, default_sc, callback in (
            ("launch_training", "Launch training", "F5", self.launch_training),
            ("stop_run", "Stop", "Shift+F5", self.stop_run),
            ("add_to_queue", "Add to queue", "Ctrl+Shift+Q", self.add_to_queue),
            ("toggle_queue", "Start or pause queue", "Ctrl+Shift+R",
             lambda: self.set_queue_running(not self.queue_running)),
            ("start_demo", "Start simulated demo", "Ctrl+F5", self.start_demo),
        ):
            action = QAction(title, self)
            sc = self.settings_manager.get("shortcuts", aid, default=default_sc) if hasattr(self, "settings_manager") else default_sc
            if sc:
                action.setShortcut(sc)
            action.triggered.connect(callback)
            run_menu.addAction(action)
            self.menu_actions[aid] = action
        view_menu = self.menuBar().addMenu("View")
        self.themes_menu = view_menu.addMenu("Themes")
        self.themes_menu.aboutToShow.connect(self.populate_themes_menu)
        view_menu.addAction("Theme builder…", self.show_theme_builder)
        view_menu.addAction("Components", lambda: self.tabs.setCurrentWidget(self.components_panel))
        view_menu.addAction("Experiment config", lambda: self.tabs.setCurrentWidget(self.config_panel))
        view_menu.addAction("Workflows", lambda: self.tabs.setCurrentWidget(self.workflows_panel))
        view_menu.addAction("Training monitor", lambda: self.tabs.setCurrentWidget(self.monitor_panel))
        view_menu.addAction("Results browser", lambda: self.tabs.setCurrentWidget(self.results_panel))
        view_menu.addAction("Plot viewer", lambda: self.tabs.setCurrentWidget(self.plot_viewer))
        view_menu.addAction("TensorBoard", self.show_tensorboard)
        view_menu.addAction("Job queue", lambda: self.tabs.setCurrentWidget(self.queue_panel))
        view_menu.addAction("Terminal", lambda: self.tabs.setCurrentWidget(self.terminal_panel))
        view_menu.addAction("Console", lambda: self.tabs.setCurrentWidget(self.console_panel))
        view_menu.addAction("Settings & About", lambda: self.tabs.setCurrentWidget(self.settings_panel))
        view_menu.addAction("Restore default layout", lambda: self.restoreState(self.default_layout))
        self.auto_hide_action = QAction("Auto-hide sidebar", self, checkable=True)
        self.auto_hide_action.setChecked(self.tabs.auto_hide)
        self.auto_hide_action.toggled.connect(lambda on: self.tabs.set_auto_hide(on))
        view_menu.addAction(self.auto_hide_action)
        view_menu.addSeparator()
        view_menu.addAction("Preferences: Open Settings File", self.open_settings_file)
        help_menu = self.menuBar().addMenu("Help")
        help_menu.addAction("Settings & About", self.show_about)

    def open_settings_file(self):
        """Open the workspace settings.toml in the user's default text editor."""
        path = self.settings_manager.workspace_settings_path
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
        self.statusBar().showMessage(f"Opened {path.name} — changes apply automatically on save.", 6000)

    def show_tensorboard(self):
        self.tabs.setCurrentWidget(self.tensorboard_panel)

    def tab_changed(self, index):
        if self.tabs.widget(index) is self.tensorboard_panel:
            self.tensorboard_panel.activate()
        if self.tabs.widget(index) is self.queue_panel:
            self.poll_queue()
            self.queue_timer.start()
        if hasattr(self, "terminal_panel") and self.tabs.widget(index) is self.terminal_panel:
            self.terminal_panel.terminal.focus_terminal()
            self.terminal_panel.terminal.fit_terminal()
        if hasattr(self, "console_panel") and self.tabs.widget(index) is self.console_panel:
            self.command.setFocus()
        if hasattr(self, "settings_panel") and self.tabs.widget(index) is self.settings_panel:
            self.test_backend_connection()
            if hasattr(self, "settings_view"):
                self.settings_view.reset_to_ascii()

    def show_about(self):
        if hasattr(self, "settings_panel"):
            self.tabs.setCurrentWidget(self.settings_panel)
            if hasattr(self, "settings_view"):
                self.settings_view.reset_to_ascii()
        else:
            dialog = AboutDialog(self)
            dialog.exec()
            dialog.deleteLater()

    def theme_changed(self):
        self.theme_status.setText(f"  ●  Local workspace    /    {self.theme_manager.active['name']}")
        self.highlighter.rehighlight()
        self.tabs.refresh_icons()
        for tree in (getattr(self, "config_tree", None),
                     getattr(getattr(self, "components_panel", None), "components_tree", None)):
            if tree is not None:
                tree.refresh_icons()
        if hasattr(self, "settings_theme_select"):
            self.settings_theme_select.blockSignals(True)
            self.settings_theme_select.setCurrentText(self.theme_manager.active["name"])
            self.settings_theme_select.blockSignals(False)
        if hasattr(self, "settings_ascii"):
            self.settings_ascii.update()
        if hasattr(self, "settings_backend_status"):
            self.test_backend_connection()
        if hasattr(self, "terminal_panel"):
            self.terminal_panel.apply_theme(self.theme_manager.active)
        if hasattr(self, "settings_view"):
            self.settings_view.refresh_styles()
        if self.isVisible():  # before the first show, showEvent colours the title bar
            apply_title_bar(self, self.theme_manager.active["colors"])

    def showEvent(self, event):
        super().showEvent(event)
        apply_title_bar(self, self.theme_manager.active["colors"])  # the native window exists once shown

    def populate_themes_menu(self):
        self.themes_menu.clear()
        for name, theme in self.theme_manager.themes().items():
            action = self.themes_menu.addAction(name)
            action.setCheckable(True)
            action.setChecked(name == self.theme_manager.active["name"])
            action.triggered.connect(lambda _, palette=theme: self.select_theme(palette))
        self.themes_menu.addSeparator()
        self.themes_menu.addAction("Theme builder…", self.show_theme_builder)

    def persist_theme_choice(self, theme_name):
        """Record a chosen theme in settings.toml. Every way of picking a theme (View menu, Settings
        dropdown, theme builder) ends here, so a later settings reload can't revert to a stale name."""
        if hasattr(self, "settings_manager") and self.settings_manager.theme != theme_name:
            self.settings_manager.set("appearance", "theme", theme_name)

    def select_theme(self, theme):
        try:
            self.theme_manager.commit(theme)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Could not save theme", str(exc))

    def show_theme_builder(self):
        dialog = ThemeBuilder(self.theme_manager, self)
        dialog.exec()
        dialog.deleteLater()

    def config(self):
        if hasattr(self, "config_viewer") and getattr(self.config_viewer, "raw_data", None):
            d = self.config_viewer.raw_data
            exp_id = d.get("experiment_id") or (self.config_viewer.current_path.stem if getattr(self.config_viewer, "current_path", None) else "cartpole_ppo_baseline")
            seed = int(d.get("seed") if d.get("seed") is not None else 42)
            steps = int(d.get("total_timesteps") or 10000)
            tb = bool(d.get("tensorboard", True))
            methods = d.get("methods", {})
            ppo_spec = methods.get("ppo", {}) if isinstance(methods, dict) else {}
            lr = float(ppo_spec.get("lr") or 0.0003)
            batch = int(ppo_spec.get("batch_size") or 64)
            gamma = float(ppo_spec.get("gamma") or 0.99)
            return Config(name=str(exp_id), seed=seed, total_timesteps=steps, lr=lr, batch_size=batch, gamma=gamma, tensorboard=tb)
        return Config()

    def update_config(self, *_):
        self.compose_timer.start()

    def request_compose(self):
        self.compose_serial += 1
        serial, config = self.compose_serial, self.config()
        exp_target = getattr(self, "current_experiment", None) or BASE_EXPERIMENT
        overrides = self.config_viewer.get_overrides() if hasattr(self, "config_viewer") else config.overrides()
        payload = {"experiment": exp_target, "overrides": overrides}
        self.backend.post("/api/config/compose", payload,
                          lambda data, error: self.compose_finished(serial, config, data, error))

    def compose_finished(self, serial, config, data, error):
        if serial != self.compose_serial:
            return  # superseded by a newer edit
        self.compose_valid = bool(not error and isinstance(data, dict) and data.get("valid", False))
        self.update_launch_state()
        if error or not isinstance(data, dict):
            self.set_backend_state(False, error or "Invalid backend response")
            self.preview_status.setText("Backend offline • local recipe draft, not validated")
            self.preview_errors.hide()
            self.builder_status.setObjectName("muted")
            self.builder_status.setText("Backend offline: this config has not been validated.")
            self.builder_status.setStyle(self.builder_status.style())
            self.preview.setPlainText(config.recipe_yaml())
            return
        self.set_backend_state(True)
        if self.schema is None and not self.schema_pending:
            self.request_schema(apply_defaults=False)
        header = ["# Resolved by the backend exactly as the command line would:",
                  "#   " + " ".join(data.get("argv", []))]
        header += [f"# Notice: {notice}" for notice in data.get("notices", [])]
        sections = ["\n".join(header)]
        if data.get("methods"):
            sections.append("# Methods — the settings each method actually trains with\nmethods:\n"
                            + "".join(f"  {line}\n" for line in data["methods_yaml"].splitlines()).rstrip())
        sections.append("# Full composed config — agent.* holds defaults that the method settings above override\n"
                        + data["config_yaml"])
        self.preview.setPlainText("\n\n".join(sections))
        if data["valid"]:
            self.preview_status.setText("Resolved config • ✓ valid for the training pipeline")
            self.preview_errors.hide()
            self.builder_status.setObjectName("configOk")
            rollouts = [plan["rollout"] for plan in data.get("methods", {}).values() if plan.get("rollout")]
            detail = ""
            if rollouts and rollouts[0]["timesteps"] != config.total_timesteps:
                r = rollouts[0]
                detail = f" · trains {r['timesteps']:,} steps ({r['rollouts']} PPO rollouts × {r['size']})"
            self.builder_status.setText("✓ Config validated by backend" + detail)
        else:
            messages = [f"[{e['stage']}] {e['message']}" for e in data["errors"]]
            self.preview_status.setText(f"Resolved config • ✗ {len(messages)} error(s)")
            self.preview_errors.setText("\n".join(messages))
            self.preview_errors.show()
            self.builder_status.setObjectName("configError")
            self.builder_status.setText("✗ " + messages[0].splitlines()[0])
        self.builder_status.setStyle(self.builder_status.style())

    def set_backend_state(self, connected, error=None):
        if connected:
            self.backend_label.setText(f"BACKEND   ●   {self.backend.base_url}  ")
            self.backend_label.setToolTip("Configs are composed and validated by the Theta-IDE backend API.")
        else:
            self.backend_label.setText("BACKEND   ○   offline  ")
            self.backend_label.setToolTip(f"{self.backend.base_url}: {error}\nStart it from the repository root with:\n"
                                          "uvicorn src.app.api.app:app --host 127.0.0.1 --port 8000")

    def request_schema(self, apply_defaults):
        self.schema_pending = True
        self.backend.get("/api/config/schema", lambda data, error: self.schema_loaded(data, error, apply_defaults))

    def schema_loaded(self, data, error, apply_defaults):
        self.schema_pending = False
        if error:
            self.set_backend_state(False, error)
            self.log(f"Backend not reachable at {self.backend.base_url} ({error}). "
                     "Start it with: uvicorn src.app.api.app:app --host 127.0.0.1 --port 8000")
            return
        self.schema = data
        env = data["environment"]
        method = data["method"]
        for title, text in (("Environment", env.get("env_id") or env["name"]),
                            ("Method", f"{method['agent'].upper()} · {method['model']}"),
                            ("Training mode", f"Online · {data['paradigm']}")):
            if title in self.fixed_fields:
                self.fixed_fields[title].setItemText(0, text)
        for field in data["fields"]:
            widget = self.fields.get(field["key"])
            if widget is None:
                continue
            widget.setToolTip(f"{field['help']}\nOverride: {field['key']}")
            if "choices" in field:
                current = widget.currentText()
                widget.blockSignals(True)
                widget.clear()
                widget.addItems([str(choice) for choice in field["choices"]])
                widget.setCurrentText(current)
                widget.blockSignals(False)
            elif "min" in field:
                widget.setRange(field["min"], field["max"])
                if "step" in field:
                    widget.setSingleStep(field["step"])
            if apply_defaults and field["key"] != "experiment_id" and field.get("default") is not None:
                if "choices" in field:
                    widget.setCurrentText(str(field["default"]))
                elif field["type"] == "bool":
                    widget.setChecked(bool(field["default"]))
                else:
                    widget.setValue(field["default"])
        self.log(f"Experiment builder loaded from backend: {env['name']} / {method['agent']} ({data['base_experiment']}).")
        self.update_config()

    def set_config(self, config):
        if hasattr(self, "config_viewer"):
            d = self.config_viewer.raw_data
            d["experiment_id"] = config.name
            d["seed"] = config.seed
            d["total_timesteps"] = config.total_timesteps
            d["tensorboard"] = config.tensorboard
            methods = d.setdefault("methods", {})
            if isinstance(methods, dict):
                ppo_spec = methods.setdefault("ppo", {})
                if isinstance(ppo_spec, dict):
                    ppo_spec["lr"] = config.lr
                    ppo_spec["batch_size"] = config.batch_size
                    ppo_spec["gamma"] = config.gamma
            self.config_viewer._render_boxes()
        self.update_config()

    def new_experiment(self):
        if hasattr(self, "config_tree"):
            self.config_tree.prompt_new_in_group()
        else:
            self.set_config(Config(name="cartpole_ppo_experiment"))
            self.tabs.setCurrentWidget(self.config_panel)

    def persist(self, run):
        try:
            self.store.save(run)
            return True
        except OSError as exc:
            self.log(f"SAVE FAILED: {exc}")
            self.statusBar().showMessage(f"Could not save run: {exc}", 12000)
            return False

    def update_launch_state(self):
        idle = self.active is None
        ready = idle and self.compose_valid
        self.start_button.setEnabled(ready)
        self.queue_button.setEnabled(self.compose_valid)
        self.stop_button.setEnabled(not idle)
        if not idle:
            tip = "A run is active. Stop it before launching another."
        elif not self.compose_valid:
            tip = "Launch needs the backend running and a config it has validated."
        else:
            tip = "Train the builder's config on this machine through the backend (F5)"
        self.start_button.setToolTip(tip)
        run = self.active
        if run is None:
            self.state_label.setText("TRAINING   •   idle  ")
        elif run["simulated"]:
            self.state_label.setText(f"DEMO   •   {run['status']}  ")
        else:
            self.state_label.setText(f"TRAINING   •   {run['status']}   •   {run['backend']['experiment_id']}  ")

    def activate(self, run):
        self.active = run
        if run not in self.runs:
            self.runs.insert(0, run)
        self.select_run(run)
        self.refresh_runs()
        self.tabs.setCurrentWidget(self.monitor_panel)
        self.update_launch_state()

    # ── Live training through the backend ────────────────────────────────────

    def launch_training(self):
        if self.active:
            self.statusBar().showMessage("A run is already active. Stop it before launching another.", 4000)
            return
        if not self.compose_valid:
            self.statusBar().showMessage("Start the backend and fix config errors before launching.", 5000)
            return
        if hasattr(self, "config_viewer") and self.config_viewer.is_dirty:
            self.config_viewer.save_to_disk()

        config = self.config()
        exp_target = getattr(self, "current_experiment", None) or BASE_EXPERIMENT
        experiment_id = self.unique_experiment_id(config.name)
        overrides = self.config_viewer.get_overrides() if hasattr(self, "config_viewer") else config.overrides()
        overrides = [o for o in overrides if not o.startswith("++experiment_id=")] + [f"++experiment_id='{experiment_id}'"]
        self.start_button.setEnabled(False)
        self.log(f"Launching {experiment_id}:\n  python run_pipeline.py {exp_target} {' '.join(overrides)}")
        self.backend.post("/api/experiments/launch", {"experiment": exp_target, "overrides": overrides},
                          lambda data, error: self.launch_finished(config, data, error))

    def launch_finished(self, config, job, error):
        if error:
            self.log(f"Launch failed: {error}")
            self.statusBar().showMessage("Launch failed; see the console.", 8000)
            self.update_launch_state()
            return
        run = new_run(config, simulated=False)
        run.update(status="starting", log_since=0, metrics_since=0, metrics_byte=0, backend={
            key: job.get(key) for key in ("job_id", "experiment", "group", "experiment_id", "agents", "total_timesteps",
                                          "effective_timesteps")})
        if not self.persist(run):
            return
        self.activate(run)
        self.log(f"Job {job['job_id']} started. Metrics: results/logs/{job['group']}/{job['experiment_id']}/")
        self.poll_timer.start()
        self.poll_job()

    def reconnect_live_run(self):
        """Resume monitoring a run that was still training when the app last closed."""
        run = next((r for r in self.runs if not r["simulated"] and r["status"] in LIVE_STATUSES), None)
        if run:
            self.log(f"Reconnecting to {run['backend']['experiment_id']} (job {run['backend']['job_id']})…")
            self.activate(run)
            self.poll_timer.start()
        if any(r["status"] == "queued" for r in self.runs):
            self.queue_timer.start()

    def poll_job(self):
        run = self.active
        if self.poll_busy or not run or run["simulated"]:
            return
        self.poll_busy = True
        job_id = run["backend"]["job_id"]
        log_since = run.get("log_since", 0)
        byte_since = run.get("metrics_byte", 0)
        self.backend.get(f"/api/experiments/{job_id}/telemetry?since_log={log_since}&since_byte={byte_since}",
                         lambda data, error: self.telemetry_received(run, data, error))

    def telemetry_received(self, run, data, error):
        self.poll_busy = False
        if run is not self.active:
            return
        if error:
            if "404" in error:  # the backend restarted and no longer knows this job
                self.log("The backend no longer tracks this job. Metrics on disk are kept.")
                self.finish("interrupted")
            else:
                self.set_backend_state(False, error)
            return
        self.set_backend_state(True)
        job = data.get("job", {})
        for line in job.get("log", []):
            self.log(f"│ {line}")
            if "Auto-Generating" in line:
                run["phase"] = "plotting"
        run["log_since"] = job.get("log_total", run.get("log_since", 0))
        status = {"pending": "starting"}.get(job.get("status"), job.get("status"))
        if run["status"] != status:
            run["status"] = status
            self.update_launch_state()
            self.refresh_runs()

        # Update metrics
        agents = data.get("agents", {})
        agent_names = run["backend"].get("agents", [])
        if not agent_names and agents:
            agent_names = list(agents.keys())
            run["backend"]["agents"] = agent_names
        if agent_names:
            had_new = False
            for ag_name in agent_names:
                agent = agents.get(ag_name, {})
                if agent.get("reset"):
                    run["metrics"] = []
                    run["metrics_byte"] = 0
                new_rows = agent.get("rows", [])
                if new_rows:
                    run["metrics"].extend(metric_points(new_rows))
                    run["metrics_byte"] = agent.get("next_byte", run.get("metrics_byte", 0))
                    had_new = True
            if had_new and self.selected is run:
                self.render_run()

        # Live hardware telemetry badge
        hw = data.get("hardware", {})
        if hw and hw.get("cpu_percent") is not None:
            self.monitor_note.setText(f"System: CPU {hw['cpu_percent']:.1f}%  •  Process RAM {hw.get('memory_mb', 0):.0f} MB")

        if job.get("status") in FINAL_STATUSES:
            detail = f" (exit code {job.get('returncode')})" if job.get("returncode") not in (None, 0) else ""
            self.finish(job.get("status", "completed"), detail)
        elif len(run.get("metrics", [])) and run["status"] == "running":
            self.persist(run)

    # ── Job queue ────────────────────────────────────────────────────────────

    def unique_experiment_id(self, name):
        """<name>_<date>-<time>, with a counter if several jobs are created within the same second."""
        base = f"{name}_{datetime.now():%Y%m%d-%H%M%S}"
        experiment_id, n = base, 2
        while experiment_id in self.issued_ids:
            experiment_id, n = f"{base}-{n}", n + 1
        self.issued_ids.add(experiment_id)
        return experiment_id

    def add_to_queue(self):
        if not self.compose_valid:
            self.statusBar().showMessage("Start the backend and fix config errors before queueing.", 5000)
            return
        if hasattr(self, "config_viewer") and self.config_viewer.is_dirty:
            self.config_viewer.save_to_disk()

        config = self.config()
        exp_target = getattr(self, "current_experiment", None) or BASE_EXPERIMENT
        experiment_id = self.unique_experiment_id(config.name)
        overrides = self.config_viewer.get_overrides() if hasattr(self, "config_viewer") else config.overrides()
        overrides = [o for o in overrides if not o.startswith("++experiment_id=")] + [f"++experiment_id='{experiment_id}'"]
        self.queue_button.setEnabled(False)
        self.log(f"Queuing {experiment_id}:\n  python run_pipeline.py {exp_target} {' '.join(overrides)}")
        self.backend.post("/api/experiments/launch",
                          {"experiment": exp_target, "overrides": overrides, "queue": True},
                          lambda data, error: self.queued(config, data, error))

    def queued(self, config, job, error):
        if error:
            self.log(f"Could not add to queue: {error}")
            self.statusBar().showMessage("Could not add to queue; see the console.", 8000)
            return
        run = new_run(config, simulated=False)
        run.update(status="queued", log_since=0, metrics_since=0, queue_position=job.get("position"), backend={
            key: job.get(key) for key in ("job_id", "experiment", "group", "experiment_id", "agents", "total_timesteps",
                                          "effective_timesteps")})
        if not self.persist(run):
            return
        self.runs.insert(0, run)
        self.refresh_runs()
        self.queue_running = bool(job.get("queue_running"))
        place = f"position {job['position'] + 1}"
        hint = ("it runs after the jobs ahead of it" if self.queue_running
                else "the queue is paused; press Start queue in the Queue tab to begin")
        self.log(f"Queued {job['experiment_id']} ({place}): {hint}.")
        self.statusBar().showMessage(f"Added {job['experiment_id']} to the queue ({place}); {hint}.", 8000)
        self.queue_timer.start()
        self.poll_queue()

    def poll_queue(self):
        if self.queue_busy:
            return
        self.queue_busy = True
        self.backend.get("/api/queue", self.queue_received)

    def queue_received(self, data, error):
        self.queue_busy = False
        if error:
            self.queue_panel.set_offline(error)
            return
        if self.queue_running != data["running"]:
            self.queue_running = data["running"]
            self.render_run()
        self.queue_panel.set_data(data)
        where = {job["job_id"]: ("active", job) for job in data.get("active", []) if isinstance(job, dict) and "job_id" in job}
        where.update({job["job_id"]: ("queued", job) for job in data.get("queued", []) if isinstance(job, dict) and "job_id" in job})
        where.update({job["job_id"]: ("finished", job) for job in data.get("finished", []) if isinstance(job, dict) and "job_id" in job})
        changed = False
        for run in [r for r in self.runs if r["status"] == "queued"]:
            kind, job = where.get(run["backend"]["job_id"], (None, None))
            if kind == "queued":
                changed |= run.get("queue_position") != job["position"]
                run["queue_position"] = job["position"]
            elif kind == "active" and self.active is None:
                run["status"] = "starting"
                self.log(f"Queue: {run['backend']['experiment_id']} started.")
                self.persist(run)
                self.activate(run)
                self.poll_timer.start()
                self.poll_job()
            elif kind == "finished":  # removed from the queue, or finished while another run was followed
                run["status"] = job["status"]
                self.persist(run)
                changed = True
            elif kind is None:  # not in the listing (it keeps only recent finished jobs): ask about this job
                job_id = run["backend"]["job_id"]
                self.backend.get(f"/api/experiments/{job_id}/status",
                                 lambda status, err, run=run: self.queued_job_status(run, status, err))
        if changed:
            self.refresh_runs()
            self.render_run()
        # Keep watching while jobs wait or the queue runs, so its self-pause after draining is seen too
        waiting = any(r["status"] == "queued" for r in self.runs)
        if not waiting and not self.queue_running and self.tabs.currentWidget() is not self.queue_panel:
            self.queue_timer.stop()

    def queued_job_status(self, run, status, error):
        if run["status"] != "queued":
            return
        if error and "404" in error:
            run["status"] = "interrupted"  # the backend restarted and lost its queue
            self.log(f"The backend no longer has queued job {run['backend']['experiment_id']} (it was restarted).")
        elif not error and status["status"] in FINAL_STATUSES:
            run["status"] = status["status"]
        else:
            return
        self.persist(run)
        self.refresh_runs()
        self.render_run()

    def set_queue_running(self, running):
        action = "start" if running else "pause"
        self.backend.post(f"/api/queue/{action}", {}, lambda data, error: self.queue_toggled(action, data, error))

    def queue_toggled(self, action, data, error):
        if error:
            self.log(f"Could not {action} the queue: {error}")
            return
        if action == "start" and not data["running"]:
            self.statusBar().showMessage("The queue is empty; add experiments first.", 5000)
        else:
            self.log("Queue started: jobs will train one after another." if data["running"] else
                     "Queue paused: no new job will start; one already training keeps running.")
        self.queue_timer.start()
        self.poll_queue()

    def move_queued(self, job_id, position):
        self.backend.post(f"/api/queue/{job_id}/move", {"position": position},
                          lambda data, error: self.log(f"Move failed: {error}") if error else self.poll_queue())

    def remove_queued(self, job_id):
        self.backend.post(f"/api/experiments/{job_id}/cancel", {},
                          lambda data, error: self.log(f"Remove failed: {error}") if error else self.poll_queue())

    def open_job(self, job_id):
        run = next((r for r in self.runs if not r["simulated"] and r["backend"]["job_id"] == job_id), None)
        if run is None:
            self.statusBar().showMessage("That job was queued from another client; it has no record here.", 5000)
            return
        self.select_run(run)
        self.tabs.setCurrentWidget(self.monitor_panel)

    # ── Simulated demo (no backend needed) ───────────────────────────────────

    def start_demo(self):
        if self.active:
            self.statusBar().showMessage("A run is already active. Stop it before starting another.", 4000)
            return
        run = new_run(self.config())
        if not self.persist(run):
            return
        self.activate(run)
        self.log(f"[demo] Started {run['config']['name']} · seed {run['config']['seed']} · {run['id']}")
        self.timer.start()

    def tick(self):
        if not self.active or not self.active["simulated"]:
            return
        run = self.active
        metric = sample(Config(**run["config"]), len(run["metrics"]) + 1)
        run["metrics"].append(metric)
        self.plot_viewer.update_runs(self.runs)
        if self.selected is run:
            self.render_run()
        if len(run["metrics"]) % 10 == 0:
            self.log(f"[demo] step={metric['step']:>6}  reward={metric['reward']:.2f}  loss={metric['loss']:.4f}")
            self.persist(run)
        if len(run["metrics"]) >= 60:
            self.finish("completed")

    # ── Shared lifecycle ─────────────────────────────────────────────────────

    def stop_run(self):
        run = self.active
        if not run:
            return
        if run["simulated"]:
            self.finish("stopped")
            return
        self.stop_button.setEnabled(False)
        self.log(f"Stopping {run['backend']['experiment_id']}: terminating the pipeline and its training processes…")
        self.backend.post(f"/api/experiments/{run['backend']['job_id']}/cancel", {},
                          lambda data, error: self.log(f"Stop failed: {error}") if error else None)
        # The next status poll reports "cancelled" and finishes the run.

    def finish(self, status, detail=""):
        self.timer.stop()
        self.poll_timer.stop()
        run = self.active
        run["status"] = status
        self.persist(run)
        prefix = "[demo] " if run["simulated"] else ""
        self.log(f"{prefix}{status.capitalize()}{detail} · {run['id']} · config and metrics saved locally")
        if status == "failed":
            self.statusBar().showMessage("Training failed; the console shows the pipeline output.", 10000)
        self.active = None
        self.update_launch_state()
        self.refresh_runs()
        self.render_run()
        if any(r["status"] == "queued" for r in self.runs):
            self.poll_queue()  # follow the next queued job as soon as it starts

    def select_run(self, run):
        self.selected = run
        if hasattr(self, "run_combo"):
            self.run_combo.blockSignals(True)
            for idx in range(self.run_combo.count()):
                if self.run_combo.itemData(idx) == run["id"]:
                    self.run_combo.setCurrentIndex(idx)
                    break
            self.run_combo.blockSignals(False)
        if hasattr(self, "btn_prev_run") and hasattr(self, "btn_next_run") and hasattr(self, "run_combo"):
            idx = self.run_combo.currentIndex()
            self.btn_prev_run.setEnabled(idx > 0)
            self.btn_next_run.setEnabled(idx >= 0 and idx < self.run_combo.count() - 1)
        self.render_run()
        self.backfill_metrics(run)

    def on_smoothing_changed(self, value):
        self.smoothing_label.setText(f"Smoothing: {value}%")
        factor = value / 100.0
        self.reward_chart.set_smoothing(factor)
        self.loss_chart.set_smoothing(factor)

    def on_run_combo_changed(self, index):
        if index < 0:
            return
        run_id = self.run_combo.itemData(index)
        if not run_id:
            return
        target_run = self.by_id(run_id)
        if target_run and target_run != self.selected:
            self.select_run(target_run)

    def select_prev_run(self):
        if not hasattr(self, "run_combo"):
            return
        idx = self.run_combo.currentIndex()
        if idx > 0:
            self.run_combo.setCurrentIndex(idx - 1)

    def select_next_run(self):
        if not hasattr(self, "run_combo"):
            return
        idx = self.run_combo.currentIndex()
        if idx < self.run_combo.count() - 1:
            self.run_combo.setCurrentIndex(idx + 1)

    def jump_to_live_run(self):
        if self.active:
            self.select_run(self.active)

    def toggle_pin_baseline(self):
        if not self.selected:
            return
        if self.pinned_baseline_run is self.selected:
            self.pinned_baseline_run = None
        else:
            self.pinned_baseline_run = self.selected
        self.render_run()

    def sync_metric_select(self, run):
        """Offer the second-chart metrics this run has; keep the user's choice when available."""
        available = [key for key in SECOND_METRICS if key in available_metrics(run)] or ["loss"]
        chosen = self.second_metric if self.second_metric in available else available[0]
        self.metric_select.blockSignals(True)
        self.metric_select.clear()
        for key in available:
            self.metric_select.addItem(SECOND_METRICS[key][1], key)
        self.metric_select.setCurrentIndex(available.index(chosen))
        self.metric_select.blockSignals(False)
        self.loss_chart.position_corner()
        return chosen

    def second_metric_changed(self, index):
        key = self.metric_select.itemData(index)
        if key:
            self.second_metric = key
            self.render_run()

    def backfill_metrics(self, run):
        """Fetch the full metrics of a finished trained run recorded before all metrics were kept."""
        if run["simulated"] or run["status"] not in FINAL_STATUSES or run.get("metrics_backfilled"):
            return
        if not run["metrics"] or any("entropy" in m for m in run["metrics"]):
            return
        backend = run["backend"]
        agent = (backend.get("agents") or ["ppo"])[0]
        path = f"/api/runs/{backend['group']}/{backend['experiment_id']}/{agent}/metrics"
        self.backend.get(path, lambda data, error: self.backfill_received(run, data, error))

    def backfill_received(self, run, data, error):
        run["metrics_backfilled"] = True  # try once; the results folder may have been deleted
        if error:
            return
        rows = []
        for row in data["metrics"]:
            numeric = {}
            for key, value in row.items():
                try:
                    numeric[key] = float(value)
                except (TypeError, ValueError):
                    pass
            rows.append(numeric)
        points = metric_points(rows)
        if points:
            run["metrics"] = points
            self.persist(run)
            if self.selected is run:
                self.render_run()

    def render_run(self):
        run = self.selected
        if not run:
            return
        config = run["config"]
        metrics = run["metrics"]
        live = not run["simulated"]
        title = run["backend"]["experiment_id"] if live else config["name"]
        source = f"LIVE · job {run['backend']['job_id'][:8]}" if live else "SIMULATED"
        self.run_title.setText(title)
        self.run_caption.setText(f"CartPole-v1  /  PPO  /  seed {config['seed']}   •   {run['status'].upper()}   •   {source}")
        if hasattr(self, "storage_label"):
            if live:
                where = f"results/logs/{run['backend']['group']}/{run['backend']['experiment_id']}/"
                self.storage_label.setText(f"{where}  ·  results/jobs/jobs.db")
            else:
                self.storage_label.setText("Simulated demo run (in-memory)  ·  results/jobs/jobs.db")

        # Update Live Jump button
        if hasattr(self, "btn_live_jump"):
            if self.active and self.selected != self.active:
                live_id = self.active["backend"]["experiment_id"] if not self.active["simulated"] else self.active["config"]["name"]
                chip_text = f"Jump to live ({live_id[:14]}…)" if len(live_id) > 14 else f"Jump to live ({live_id})"
                self.btn_live_jump.setText(chip_text)
                self.btn_live_jump.show()
            else:
                self.btn_live_jump.hide()

        # Update Pin Baseline button
        if hasattr(self, "btn_pin_baseline"):
            if self.pinned_baseline_run is None:
                self.btn_pin_baseline.setChecked(False)
                self.btn_pin_baseline.setText("Pin as baseline")
                self.btn_pin_baseline.setToolTip("Pin this run's curve to overlay as a dashed baseline when inspecting other runs")
            elif self.pinned_baseline_run is run:
                self.btn_pin_baseline.setChecked(True)
                self.btn_pin_baseline.setText("Pinned baseline")
                self.btn_pin_baseline.setToolTip("This run is currently pinned as the baseline. Click to unpin.")
            else:
                self.btn_pin_baseline.setChecked(False)
                p_name = self.pinned_baseline_run["backend"]["experiment_id"] if not self.pinned_baseline_run["simulated"] else self.pinned_baseline_run["config"]["name"]
                label_name = (p_name[:12] + "…") if len(p_name) > 12 else p_name
                self.btn_pin_baseline.setText(f"Replace baseline ({label_name})")
                self.btn_pin_baseline.setToolTip(f"Baseline '{p_name}' is pinned. Click to replace it with this run.")

        # Update Prev/Next button states
        if hasattr(self, "btn_prev_run") and hasattr(self, "btn_next_run") and hasattr(self, "run_combo"):
            idx = self.run_combo.currentIndex()
            self.btn_prev_run.setEnabled(idx > 0)
            self.btn_next_run.setEnabled(idx >= 0 and idx < self.run_combo.count() - 1)

        reward = latest(run, "reward")
        second = self.sync_metric_select(run)
        card_title, chart_title, subtitle, fmt = SECOND_METRICS[second]
        second_value = latest(run, second)
        step = max((m["step"] for m in metrics), default=0)
        requested = config["total_timesteps"]
        # PPO trains whole rollouts, so the real budget can exceed the requested one (10,000 -> 10,240).
        # Records from before the backend reported it fall back to the steps actually run.
        budget = (run["backend"].get("effective_timesteps") or max(step, requested)) if live else requested
        self.reward_card.value.setText("—" if reward is None else f"{reward:.1f}")
        self.loss_card.title.setText(card_title)
        self.loss_card.value.setText("—" if second_value is None else format(second_value, fmt))
        self.loss_card.subtitle.setText(subtitle if live else "synthetic training loss")
        self.steps_card.value.setText(f"{step:,}")
        self.progress.setValue(min(100, round(100 * step / max(1, budget))))
        if live:
            self.reward_card.subtitle.setText("evaluation / mean episode reward")
            rounding = f"\n{requested:,} requested, rounded up to whole PPO rollouts" if budget != requested else ""
            self.steps_card.subtitle.setText(f"of {budget:,} environment steps{rounding}")
            where = f"results/logs/{run['backend']['group']}/{run['backend']['experiment_id']}/"
            if run["status"] == "queued":
                position = run.get("queue_position")
                ahead = f"position {position + 1}" if position is not None else "waiting"
                note = (f"In the job queue ({ahead}). Queued experiments train one at a time; "
                        "this view follows it when it starts.")
                if not self.queue_running:
                    note += " The queue is paused: press Start queue in the Queue tab."
            elif run["status"] == "starting" or (run["status"] == "running" and not metrics):
                note = "Starting the pipeline — the first evaluation usually appears after about 15 seconds."
            elif run["status"] == "running" and run.get("phase") == "plotting":
                note = f"Training finished — the pipeline is generating plots in results/plots/{run['backend']['group']}/…"
            elif run["status"] in FINAL_STATUSES:
                note = f"Final metrics from {where}"
                if run["status"] == "completed":
                    note += f" · plots in results/plots/{run['backend']['group']}/{run['backend']['experiment_id']}/"
            else:
                note = f"Live metrics from {where} · updated every second while training."
            self.monitor_note.setText(f"●  {note}")
        else:
            self.reward_card.subtitle.setText("synthetic evaluation / mean")
            self.steps_card.subtitle.setText("configured training budget")
            self.monitor_note.setText("●  Selected run     •     Synthetic metrics / frontend demonstration")
        xmax = budget if live else None
        has_spread = any(m.get("reward_std") is not None for m in metrics)
        self.reward_chart.set_metric("reward", "Episode reward  ·  mean ± 1 std over evaluation episodes"
                                     if has_spread else "Episode reward")
        self.reward_chart.set_series([(title, metrics, "#b8bb26")], xmax, ENV_MAX_REWARD.get(config["env"]))
        self.loss_chart.set_metric(second, chart_title)
        self.loss_chart.set_series([(title, metrics, "#83a598")], xmax)

        # Baseline overlay
        if self.pinned_baseline_run and self.pinned_baseline_run is not run:
            b_metrics = self.pinned_baseline_run["metrics"]
            b_name = self.pinned_baseline_run["backend"]["experiment_id"] if not self.pinned_baseline_run["simulated"] else self.pinned_baseline_run["config"]["name"]
            b_title = f"{b_name} (baseline)"
            self.reward_chart.set_baseline_series([(b_title, b_metrics, "#d3869b")])
            self.loss_chart.set_baseline_series([(b_title, b_metrics, "#d3869b")])
        else:
            self.reward_chart.set_baseline_series([])
            self.loss_chart.set_baseline_series([])

    def refresh_runs(self, *_):
        self.plot_viewer.update_runs(self.runs)
        if hasattr(self, "settings_runs_count_label"):
            self.settings_runs_count_label.setText(f"Total run records:  {len(self.runs)} runs")
        query = self.search.text().lower()
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        for run in self.runs:
            config = run["config"]
            name = config["name"] if run["simulated"] else run["backend"]["experiment_id"]
            if query not in f"{name} {config['seed']} {run['status']}".lower():
                continue
            row = self.table.rowCount()
            self.table.insertRow(row)
            last_reward = latest(run, "reward")
            reward = "—" if last_reward is None else f"{last_reward:.1f}"
            source = "Simulated" if run["simulated"] else "Trained"
            for col, value in enumerate((name, str(config["seed"]), run["status"], reward, source)):
                cell = QTableWidgetItem(value)
                cell.setData(Qt.ItemDataRole.UserRole, run["id"])
                # Experiment names read best left-aligned; every other value sits centered under its label.
                cell.setTextAlignment((Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter) if col == 0
                                      else Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(row, col, cell)
        self.table.blockSignals(False)

        # Sync run_combo dropdown in Monitor
        if hasattr(self, "run_combo"):
            self.run_combo.blockSignals(True)
            self.run_combo.clear()
            for run in self.runs:
                config = run["config"]
                name = config["name"] if run["simulated"] else run["backend"]["experiment_id"]
                status = run.get("status", "")
                if status in LIVE_STATUSES:
                    icon = "● "
                elif status == "completed":
                    icon = "✓ "
                elif status in ("failed", "interrupted", "stopped"):
                    icon = "✗ "
                else:
                    icon = "○ "
                last_reward = latest(run, "reward")
                rew_str = f"  ·  {last_reward:.1f} rew" if last_reward is not None else ""
                seed_str = f" (seed {config['seed']})"
                self.run_combo.addItem(f"{icon}{name}{seed_str}{rew_str}", run["id"])

            if self.selected:
                for idx in range(self.run_combo.count()):
                    if self.run_combo.itemData(idx) == self.selected["id"]:
                        self.run_combo.setCurrentIndex(idx)
                        break
            self.run_combo.blockSignals(False)

        if hasattr(self, "btn_prev_run") and hasattr(self, "btn_next_run") and hasattr(self, "run_combo"):
            idx = self.run_combo.currentIndex()
            self.btn_prev_run.setEnabled(idx > 0)
            self.btn_next_run.setEnabled(idx >= 0 and idx < self.run_combo.count() - 1)

    def by_id(self, run_id):
        return next((run for run in self.runs if run["id"] == run_id), None)

    def view_selected_plot(self):
        if self.selected:
            self.plot_viewer.show_run(self.selected["id"])
        self.tabs.setCurrentWidget(self.plot_viewer)

    def tree_selected(self, item, _):
        pass

    def table_selected(self):
        rows = self.table.selectionModel().selectedRows()
        if len(rows) == 1:
            run = self.by_id(self.table.item(rows[0].row(), 0).data(Qt.ItemDataRole.UserRole))
            self.select_run(run)
        elif len(rows) >= 2:
            runs = [self.by_id(self.table.item(r.row(), 0).data(Qt.ItemDataRole.UserRole)) for r in rows]
            runs = [r for r in runs if r is not None]
            if runs:
                palette = ("#b8bb26", "#83a598", "#fabd2f", "#d3869b", "#8ec07c")
                series = []
                for i, r in enumerate(runs[:5]):
                    name = f"{r['config'].get('name', r['id'])} (seed {r['config'].get('seed', 0)})"
                    series.append((name, r.get("metrics", []), palette[i % len(palette)]))
                self.reward_chart.set_series(series)
                self.selected = runs[0]

    def load_selected_config(self):
        rows = self.table.selectionModel().selectedRows()
        if len(rows) != 1:
            self.statusBar().showMessage("Select exactly one run to load its saved config.", 5000)
            return
        run = self.by_id(self.table.item(rows[0].row(), 0).data(Qt.ItemDataRole.UserRole))
        self.set_config(Config(**run["config"]))
        self.tabs.setCurrentWidget(self.config_panel)
        self.log(f"Loaded exact configuration from {run['id']}. Launch to create a new run.")

    def compare(self):
        rows = self.table.selectionModel().selectedRows()
        if len(rows) != 2:
            self.statusBar().showMessage("Select exactly two runs with Ctrl + click to compare.", 5000)
            return
        runs = [self.by_id(self.table.item(index.row(), 0).data(Qt.ItemDataRole.UserRole)) for index in rows]
        dialog = QDialog(self)
        dialog.setWindowTitle("Compare experiments")
        dialog.resize(820, 660)
        layout = QVBoxLayout(dialog)
        layout.addWidget(label("Two runs. One question.", "heading"))
        chart = Chart("reward", "Episode reward", band="reward_std")
        chart.set_series([(run["config"]["name"], run["metrics"], color)
                          for run, color in zip(runs, ("#b8bb26", "#83a598"))],
                         reference=ENV_MAX_REWARD.get(runs[0]["config"]["env"]))
        layout.addWidget(chart, 1)
        for run, color in zip(runs, ("#b8bb26", "#83a598")):
            legend = label(f"●  {run['config']['name']} · seed {run['config']['seed']} · {run['id']}")
            legend.setStyleSheet(f"color: {theme_color(color)}")
            layout.addWidget(legend)
        diff = QTableWidget(len(asdict(Config())), 3)
        diff.setHorizontalHeaderLabels(["Parameter", "Run A", "Run B"])
        diff.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        diff.verticalHeader().hide()
        diff.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        configs = [asdict(Config(**run["config"])) for run in runs]  # older records lack newer fields
        for row, key in enumerate(asdict(Config())):
            for col, value in enumerate((key, configs[0][key], configs[1][key])):
                cell = QTableWidgetItem(str(value))
                if configs[0][key] != configs[1][key]:
                    from PyQt6.QtGui import QColor
                    cell.setForeground(QColor(theme_color("accent")))
                diff.setItem(row, col, cell)
        layout.addWidget(diff, 1)
        synthetic = [run["config"]["name"] for run in runs if run["simulated"]]
        note = f" Synthetic curves: {', '.join(synthetic)}." if synthetic else ""
        layout.addWidget(label("Changed parameters are highlighted." + note, "muted"))
        dialog.exec()

    def export_config(self):
        config = self.config()
        path, _ = QFileDialog.getSaveFileName(self, "Export experiment recipe", f"{config.name}.yaml", "YAML (*.yaml)")
        if path:
            try:
                if hasattr(self, "config_viewer") and getattr(self.config_viewer, "raw_data", None):
                    text = yaml.safe_dump(self.config_viewer.raw_data, sort_keys=False)
                else:
                    text = config.recipe_yaml()
                Path(path).write_text(text, encoding="utf-8")
                self.log(f"Exported configuration to {path}.")
            except OSError as exc:
                QMessageBox.warning(self, "Export failed", str(exc))

    def log(self, message):
        if hasattr(self, "console"):
            self.console.appendPlainText(message)

    def console_command(self, cmd_override=None):
        if cmd_override is not None:
            command = cmd_override.strip()
        elif hasattr(self, "command"):
            command = self.command.text().strip()
            self.command.clear()
        else:
            command = ""
        self.log(f"theta › {command}")
        if command == "clear":
            self.console.clear()
        elif command == "help":
            self.log("help    Show commands\nstatus  Show the active run\nconfig  Show the equivalent run_pipeline command\nclear   Clear output\n(Use the Terminal tab for an interactive shell with tmux support.)")
        elif command == "status":
            run = self.active
            if run is None:
                state = "idle"
            elif run["simulated"]:
                state = f"simulated demo {run['status']}"
            else:
                state = f"{run['status']} · {run['backend']['experiment_id']} · job {run['backend']['job_id']}"
            self.log(f"Training: {state} · {len(self.runs)} records")
        elif command == "config":
            self.log(self.config().command())
        elif command:
            self.log(f"Unknown command '{command}'. Type help or switch to the Terminal tab.")

    def closeEvent(self, event):
        if self.active and self.active["simulated"]:
            self.stop_run()
        elif self.active:
            self.persist(self.active)  # training keeps running in the backend; reopening reconnects
        if hasattr(self, "reloader") and self.reloader is not None:
            self.reloader.stop()
        if hasattr(self, "terminal_panel"):
            self.terminal_panel.terminal.close()
        if hasattr(self, "hotkey_manager"):
            self.hotkey_manager.cleanup()
        event.accept()

    def reload_frontend(self):
        """Reload the frontend process cleanly."""
        from .terminal_reloader import reload_frontend
        reload_frontend(window=self, app=QApplication.instance())


def main():
    parser = argparse.ArgumentParser(description="ThetaIDE PyQt frontend proof of concept")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent.parent / ".thetaide" / "runs")
    parser.add_argument("--api-url", default=os.environ.get("THETAIDE_API_URL", DEFAULT_URL),
                        help="Theta-IDE backend API (default: %(default)s)")
    parser.add_argument("--install-desktop-entry", action="store_true",
                        help="Linux: add ThetaIDE (with its icon) to the application menu and exit")
    args = parser.parse_args()
    if args.install_desktop_entry:
        entry, icon = install_desktop_entry()
        print(f"Installed {entry}\nInstalled {icon}")
        return 0
    set_windows_app_id()  # before any window: the taskbar then shows ThetaIDE's icon, not python.exe's
    if sys.platform == "darwin":
        try:
            import ctypes
            libc = ctypes.CDLL(None)
            if hasattr(libc, "setprogname"):
                libc.setprogname(b"ThetaIDE")
        except Exception:
            pass

    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)
    QApplication.setApplicationName("ThetaIDE")
    QApplication.setApplicationDisplayName("ThetaIDE")
    QApplication.setDesktopFileName(DESKTOP_FILE_NAME)
    QApplication.setOrganizationName("ThetaIDE")
    QApplication.setOrganizationDomain("thetaide.org")

    app = QApplication(["ThetaIDE"] + sys.argv[1:])
    app.setApplicationName("ThetaIDE")
    app.setApplicationDisplayName("ThetaIDE")
    app.setStyle("Fusion")
    app.setFont(QFont("Segoe UI", 10))
    app.setStyleSheet(STYLE)
    app.wheel_guard = wheel_guard.install(app)

    if ICON_PATH.is_file():
        app.setWindowIcon(app_icon())  # every window's title bar, the taskbar and (macOS) the Dock

    try:
        window = Window(args.data_dir, args.api_url)
    except OSError as exc:
        QMessageBox.critical(None, "Workspace unavailable", f"Could not open local run storage:\n{exc}")
        return 1
    window.show()

    from .terminal_reloader import TerminalReloader

    reloader = TerminalReloader(window=window, app=app)
    window.reloader = reloader
    app.aboutToQuit.connect(reloader.stop)

    # Clean POSIX signal handling: allow Ctrl+C (SIGINT) and SIGTERM to quit QApplication cleanly
    import signal

    def _sig_handler(*_):
        reloader.stop()
        app.quit()

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)
    sigint_timer = QTimer()
    sigint_timer.start(250)
    sigint_timer.timeout.connect(lambda: None)

    if sys.stdin.isatty():
        sys.stdout.write(
            "\033[90m[Theta-IDE] Ready · Press \033[1;36mCtrl+R\033[0m\033[90m in this terminal to reload frontend\033[0m\r\n"
        )
        sys.stdout.flush()

    reloader.start()

    try:
        return app.exec()
    finally:
        reloader.stop()

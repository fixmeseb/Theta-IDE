"""Training monitor panel: live and historic curves for the selected run.

The panel owns its widgets and drawing. Run state (the run list, the selected and active runs,
the pinned baseline, the queue) stays on the host window, which the rest of the app also reads.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QProgressBar, QPushButton, QSlider, QVBoxLayout, QWidget

from ..charts import Chart
from ..model import FINAL_STATUSES, LIVE_STATUSES, available_metrics, latest
from ..results.metrics_catalog import describe, group_sort_key, metric_group
from ..widgets import MetricCard, label

if TYPE_CHECKING:
    from ..app import Window

# Second-chart metrics with hand-written text: (card title, chart title, card subtitle, number format).
# Any other metric a run logged is offered too, described by the metrics catalog.
SECOND_METRICS = {
    "loss": ("TRAINING LOSS", "Total loss", "policy + 0.5 × value − 0.01 × entropy", ".4f"),
    "value_loss": ("VALUE LOSS", "Value loss", "value-function error · critic MSE loss", ".3f"),
    "policy_loss": ("POLICY LOSS", "Policy loss", "PPO clipped surrogate objective", ".4f"),
    "entropy": ("POLICY ENTROPY", "Policy entropy", "how random actions are · ln 2 ≈ 0.693 is uniform", ".3f"),
    "approx_kl": ("APPROX. KL", "Approximate KL per update", "size of each policy update", ".5f"),
}
# Highest possible evaluation reward per environment, drawn as the reward chart's ceiling.
ENV_MAX_REWARD = {"cartpole": 500}
# Shown in the reward chart, so not offered for the second one.
_REWARD_KEYS = {"reward", "reward_std"}


def second_metric_choices(run) -> list[str]:
    """Every metric this run logged except reward: the hand-described ones first, then the rest by group."""
    available = available_metrics(run) - _REWARD_KEYS
    known = [key for key in SECOND_METRICS if key in available]
    others = sorted(available - set(SECOND_METRICS), key=lambda n: (group_sort_key(metric_group(n)), n))
    return known + others or ["loss"]


def second_metric_text(key: str) -> tuple[str, str, str, str]:
    """(card title, chart title, card subtitle, number format) for any metric key."""
    if key in SECOND_METRICS:
        return SECOND_METRICS[key]
    info = describe(key)
    return info.label.upper(), info.label, info.description or key, info.fmt


def run_display_name(run) -> str:
    return run["config"]["name"] if run["simulated"] else run["backend"]["experiment_id"]


class MonitorPanel(QWidget):
    """The Training monitor tab."""

    # Widgets the host window also exposes under the same names (existing code and tests use them).
    EXPOSED = (
        "smoothing_label", "smoothing_slider", "run_combo", "btn_prev_run", "btn_next_run", "btn_live_jump",
        "btn_pin_baseline", "run_title", "run_caption", "storage_label", "reward_card", "loss_card", "steps_card",
        "progress", "reward_chart", "loss_chart", "metric_select", "monitor_note",
    )

    def __init__(self, host: Window):
        super().__init__()
        self.host = host
        self.second_metric = "loss"
        layout = QVBoxLayout(self)
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
        self.loss_chart = Chart(self.second_metric, SECOND_METRICS[self.second_metric][1], zero_based=False)
        self.metric_select = QComboBox()
        self.metric_select.setToolTip("Metric shown in this chart and the second card")
        # Re-measure when the choices change: other engines' metric names are longer than PPO's.
        self.metric_select.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.metric_select.currentIndexChanged.connect(self.second_metric_changed)
        self.loss_chart.set_corner_widget(self.metric_select)
        layout.addWidget(self.reward_chart, 3)
        layout.addWidget(self.loss_chart, 2)
        self.monitor_note = label("", "muted")
        self.monitor_note.setWordWrap(True)
        monitor_footer = QHBoxLayout()
        monitor_footer.addWidget(self.monitor_note, 1)
        monitor_footer.addWidget(host.button("Open TensorBoard", host.show_tensorboard))
        layout.addLayout(monitor_footer)

    # ── Run navigation ───────────────────────────────────────────────────────

    def set_runs(self, runs, selected):
        """Refill the run selector, keeping the selected run highlighted."""
        self.run_combo.blockSignals(True)
        self.run_combo.clear()
        for run in runs:
            config = run["config"]
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
            self.run_combo.addItem(f"{icon}{run_display_name(run)}{seed_str}{rew_str}", run["id"])
        if selected:
            self._highlight(selected)
        self.run_combo.blockSignals(False)
        self.update_nav_buttons()

    def show_selected(self, run):
        """Point the run selector at `run` without re-triggering a selection."""
        self.run_combo.blockSignals(True)
        self._highlight(run)
        self.run_combo.blockSignals(False)
        self.update_nav_buttons()

    def _highlight(self, run):
        for idx in range(self.run_combo.count()):
            if self.run_combo.itemData(idx) == run["id"]:
                self.run_combo.setCurrentIndex(idx)
                break

    def update_nav_buttons(self):
        idx = self.run_combo.currentIndex()
        self.btn_prev_run.setEnabled(idx > 0)
        self.btn_next_run.setEnabled(0 <= idx < self.run_combo.count() - 1)

    def on_run_combo_changed(self, index):
        if index < 0:
            return
        run_id = self.run_combo.itemData(index)
        if not run_id:
            return
        target_run = self.host.by_id(run_id)
        if target_run and target_run != self.host.selected:
            self.host.select_run(target_run)

    def select_prev_run(self):
        idx = self.run_combo.currentIndex()
        if idx > 0:
            self.run_combo.setCurrentIndex(idx - 1)

    def select_next_run(self):
        idx = self.run_combo.currentIndex()
        if idx < self.run_combo.count() - 1:
            self.run_combo.setCurrentIndex(idx + 1)

    def jump_to_live_run(self):
        if self.host.active:
            self.host.select_run(self.host.active)

    def toggle_pin_baseline(self):
        host = self.host
        if not host.selected:
            return
        host.pinned_baseline_run = None if host.pinned_baseline_run is host.selected else host.selected
        self.render()

    # ── Charts ───────────────────────────────────────────────────────────────

    def on_smoothing_changed(self, value):
        self.smoothing_label.setText(f"Smoothing: {value}%")
        factor = value / 100.0
        self.reward_chart.set_smoothing(factor)
        self.loss_chart.set_smoothing(factor)

    def sync_metric_select(self, run):
        """Offer every second-chart metric this run has; keep the user's choice when available."""
        available = second_metric_choices(run)
        chosen = self.second_metric if self.second_metric in available else available[0]
        self.metric_select.blockSignals(True)
        self.metric_select.clear()
        for key in available:
            self.metric_select.addItem(second_metric_text(key)[1], key)
        self.metric_select.setCurrentIndex(available.index(chosen))
        self.metric_select.blockSignals(False)
        self.loss_chart.position_corner()
        return chosen

    def second_metric_changed(self, index):
        key = self.metric_select.itemData(index)
        if key:
            self.second_metric = key
            self.render()

    def show_hardware(self, hw):
        """Live CPU and memory use of the training process, in the footer."""
        self.monitor_note.setText(f"System: CPU {hw['cpu_percent']:.1f}%  •  Process RAM {hw.get('memory_mb', 0):.0f} MB")

    def show_comparison(self, series):
        """Overlay several runs' reward curves (the Results browser's multi-selection)."""
        self.reward_chart.set_series(series)

    def render(self):
        host = self.host
        run = host.selected
        if not run:
            return
        config = run["config"]
        metrics = run["metrics"]
        live = not run["simulated"]
        title = run_display_name(run)
        if run.get("origin") == "cli":  # found in results/logs: no app job, and any environment or method
            source = "COMMAND LINE"
            setup = f"{config['env']}  /  {' + '.join(run['backend'].get('agents') or ['?']).upper()}"
        else:
            source = f"LIVE · job {run['backend']['job_id'][:8]}" if live else "SIMULATED"
            setup = "CartPole-v1  /  PPO"
        self.run_title.setText(title)
        self.run_caption.setText(f"{setup}  /  seed {config['seed']}   •   {run['status'].upper()}   •   {source}")
        if live:
            where = f"results/logs/{run['backend']['group']}/{run['backend']['experiment_id']}/"
            self.storage_label.setText(f"{where}  ·  results/jobs/jobs.db")
        else:
            self.storage_label.setText("Simulated demo run (in-memory)  ·  results/jobs/jobs.db")

        # Update Live Jump button
        if host.active and host.selected != host.active:
            live_id = run_display_name(host.active)
            chip_text = f"Jump to live ({live_id[:14]}…)" if len(live_id) > 14 else f"Jump to live ({live_id})"
            self.btn_live_jump.setText(chip_text)
            self.btn_live_jump.show()
        else:
            self.btn_live_jump.hide()

        # Update Pin Baseline button
        baseline = host.pinned_baseline_run
        if baseline is None:
            self.btn_pin_baseline.setChecked(False)
            self.btn_pin_baseline.setText("Pin as baseline")
            self.btn_pin_baseline.setToolTip("Pin this run's curve to overlay as a dashed baseline when inspecting other runs")
        elif baseline is run:
            self.btn_pin_baseline.setChecked(True)
            self.btn_pin_baseline.setText("Pinned baseline")
            self.btn_pin_baseline.setToolTip("This run is currently pinned as the baseline. Click to unpin.")
        else:
            self.btn_pin_baseline.setChecked(False)
            p_name = run_display_name(baseline)
            label_name = (p_name[:12] + "…") if len(p_name) > 12 else p_name
            self.btn_pin_baseline.setText(f"Replace baseline ({label_name})")
            self.btn_pin_baseline.setToolTip(f"Baseline '{p_name}' is pinned. Click to replace it with this run.")

        self.update_nav_buttons()

        reward = latest(run, "reward")
        second = self.sync_metric_select(run)
        card_title, chart_title, subtitle, fmt = second_metric_text(second)
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
                if not host.queue_running:
                    note += " The queue is paused: press Start queue in the Queue tab."
            elif run["status"] == "starting" or (run["status"] == "running" and not metrics):
                note = "Starting the pipeline — the first evaluation usually appears after about 15 seconds."
            elif run["status"] == "running" and run.get("phase") == "plotting":
                note = f"Training finished — the pipeline is generating plots in results/plots/{run['backend']['group']}/…"
            elif run["status"] == "incomplete":
                note = (f"Metrics from {where} · this run was started outside the app and has no recorded end "
                        "time: it may still be training, or it stopped early.")
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
        if baseline and baseline is not run:
            b_title = f"{run_display_name(baseline)} (baseline)"
            self.reward_chart.set_baseline_series([(b_title, baseline["metrics"], "#d3869b")])
            self.loss_chart.set_baseline_series([(b_title, baseline["metrics"], "#d3869b")])
        else:
            self.reward_chart.set_baseline_series([])
            self.loss_chart.set_baseline_series([])

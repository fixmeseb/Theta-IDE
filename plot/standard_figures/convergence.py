#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot.base import BasePlotter, clean_label, get_style_info, moving_average


class ConvergencePlotter(BasePlotter):
    # Composable: its output goes through draw_metric(), so the Figures
    # pane can arrange it alongside other panels on one figure.
    draws_metric_series = True

    def __init__(self):
        super().__init__("convergence")

    def run(self, exp_id: str, cli_overrides: dict | None = None):
        cfg, group, output_dir = self.get_effective_config(exp_id, cli_overrides)
        metrics = cfg.get("metrics", ["eval/reward", "train/reward", "train/length"])
        self.plot_metric_series(exp_id, group, output_dir, metrics, cfg)

"""In-process services for the desktop app: config composition, run catalog, metrics, jobs.

These used to sit behind the FastAPI server (src/app/api). The GUI now imports them directly
and runs training as a QProcess (frontend/job_manager.py), so there is no server to start.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
LOGS_DIR = PROJECT_ROOT / "results" / "logs"
PLOTS_DIR = PROJECT_ROOT / "results" / "plots"
JOBS_DIR = PROJECT_ROOT / "results" / "jobs"
TENSORBOARD_DIR = PROJECT_ROOT / "results" / "tensorboard"
EXPORTS_DIR = PROJECT_ROOT / "results" / "exports"

from .experiment_exporter import ExperimentExporter, ExportOptions, ExportResult  # noqa: E402
from .plugin_scaffolder import PluginScaffolder, ScaffoldOptions, ScaffoldResult  # noqa: E402

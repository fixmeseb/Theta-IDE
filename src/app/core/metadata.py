import logging
import os
import platform
import socket
import subprocess
import sys

logger = logging.getLogger(__name__)


def get_git_info():
    """Retrieve current git commit, branch, and dirty status.

    Returns:
        tuple: (git_commit, git_branch, git_dirty)
    """
    git_commit = None
    git_branch = None
    git_dirty = None
    try:
        git_commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, timeout=5)
            .decode("utf-8")
            .strip()
        )
        git_branch = (
            subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL, timeout=5)
            .decode("utf-8")
            .strip()
        )
        status = (
            subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, timeout=5)
            .decode("utf-8")
            .strip()
        )
        git_dirty = len(status) > 0
    except (subprocess.SubprocessError, FileNotFoundError, OSError) as e:
        logger.debug("Failed to retrieve git info: %s", e)
    except Exception as e:
        logger.debug("Unexpected error retrieving git info: %s", e)
    return git_commit, git_branch, git_dirty


def save_git_diff(output_dir, git_dirty=None):
    """Save the git diff to a file in the specified directory.

    Args:
        output_dir: The directory to save the diff.
        git_dirty: Optional boolean indicating if git tree is dirty.

    Returns:
        str or None: The filename of the saved diff, or None if no diff.
    """
    try:
        if git_dirty is None:
            _, _, git_dirty = get_git_info()
        if git_dirty:
            diff = subprocess.check_output(["git", "diff", "HEAD"], stderr=subprocess.DEVNULL, timeout=5).decode(
                "utf-8"
            )
            diff_path = os.path.join(output_dir, "git_patch.diff")
            with open(diff_path, "w") as f:
                f.write(diff)
            return "git_patch.diff"
    except (subprocess.SubprocessError, FileNotFoundError, OSError) as e:
        logger.warning("Could not save git diff to %s: %s", output_dir, e)
    except Exception as e:
        logger.warning("Unexpected error saving git diff to %s: %s", output_dir, e)
    return None


def collect_run_metadata(cfg=None):
    """Collect runtime metadata including git provenance and system info.

    Args:
        cfg: Optional hydra config object to extract seed.

    Returns:
        dict: Collected metadata dictionary.
    """
    git_commit, git_branch, git_dirty = get_git_info()
    git_diff_path = None
    if git_dirty:
        git_diff_path = "git_patch.diff"  # Will be saved by save_git_diff

    try:
        import torch

        torch_version = torch.__version__
        cuda_version = torch.version.cuda or "N/A"
    except ImportError:
        torch_version = "N/A"
        cuda_version = "N/A"

    try:
        import lightning as L

        lightning_version = L.__version__
    except ImportError:
        lightning_version = "N/A"

    seed = getattr(cfg, "seed", None) if cfg else None
    description = None
    if cfg is not None:
        description = getattr(cfg, "description", None) or (cfg.get("description") if isinstance(cfg, dict) else None)

    return {
        "git_commit": git_commit,
        "git_branch": git_branch,
        "git_dirty": git_dirty,
        "git_diff_path": git_diff_path if git_dirty else None,
        "cli_command": " ".join(sys.argv),
        "seed": seed,
        "description": description,
        "system": {
            "hostname": socket.gethostname(),
            "os": platform.platform(),
            "python": platform.python_version(),
            "torch": torch_version,
            "cuda": cuda_version,
            "lightning": lightning_version,
        },
    }

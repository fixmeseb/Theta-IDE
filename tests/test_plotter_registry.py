"""Tests for Plotter Registry, BasePlotter contract, and modular plotter discovery."""
from pathlib import Path
import pytest

from plot.base import BasePlotter
from src.app.core.plotter_registry import (
    PLOTTER_REGISTRY,
    auto_discover_plotters,
    get_plotter_class,
    list_registered_plotters,
    register_plotter,
)


def test_register_and_get_plotter():
    @register_plotter("custom_scatter_test")
    class CustomScatterPlotter(BasePlotter):
        def __init__(self):
            super().__init__("custom_scatter_test")

        def run(self, exp_id: str, cli_overrides=None):
            return "ran_custom_scatter"

    cls = get_plotter_class("custom_scatter_test")
    assert cls is CustomScatterPlotter

    inst = cls()
    assert inst.name == "custom_scatter_test"
    assert inst.run("test_exp") == "ran_custom_scatter"


def test_auto_discover_plotters():
    auto_discover_plotters()
    plotters = list_registered_plotters()

    # Core plotters must always be discoverable if standard_figures is installed
    if (Path(__file__).resolve().parent.parent / "plot" / "standard_figures").exists():
        assert "convergence" in plotters
        assert "losses" in plotters
        assert "reports" in plotters

    # Verify discovery can find plotters in installed modular subfolders
    subpkg = Path(__file__).resolve().parent.parent / "plot" / "mock_test_suite"
    subpkg.mkdir(exist_ok=True)
    try:
        (subpkg / "__init__.py").write_text("", encoding="utf-8")
        (subpkg / "mock_plot.py").write_text(
            "from plot.base import BasePlotter\n"
            "from src.app.core.plotter_registry import register_plotter\n"
            "@register_plotter('mock_suite_metric')\n"
            "class MockSuitePlotter(BasePlotter):\n"
            "    def __init__(self):\n"
            "        super().__init__('mock_suite_metric')\n",
            encoding="utf-8",
        )
        auto_discover_plotters()
        assert "mock_suite_metric" in list_registered_plotters()
    finally:
        import shutil
        shutil.rmtree(subpkg, ignore_errors=True)

    if (Path(__file__).resolve().parent.parent / "plot" / "standard_figures").exists():
        conv_cls = get_plotter_class("convergence")
        assert conv_cls is not None
        assert issubclass(conv_cls, BasePlotter)


def test_base_plotter_config_resolution(tmp_path):
    """Verify that BasePlotter searches in/config/plot/ for Tier 1 default configs."""
    cfg_dir = tmp_path / "in" / "config" / "plot"
    cfg_dir.mkdir(parents=True)
    cfg_file = cfg_dir / "test_diag.yaml"
    cfg_file.write_text("smoothing_window: 25\ndpi: 600\n", encoding="utf-8")

    class TestDiagPlotter(BasePlotter):
        def __init__(self):
            super().__init__("test_diag")

    # Mock project root to tmp_path
    import plot.base
    orig_root = plot.base.PROJECT_ROOT
    try:
        plot.base.PROJECT_ROOT = str(tmp_path)
        inst = TestDiagPlotter()
        assert inst.default_cfg.get("smoothing_window") == 25
        assert inst.default_cfg.get("dpi") == 600
    finally:
        plot.base.PROJECT_ROOT = orig_root

"""Figure workspace: compose several plots into one figure for a paper.

model.py holds the figure being built and imports no Qt, so it runs
headless in CI; sources.py bridges to the plotters in plot/; view.py is the
widget. Separate from frontend/plots.py, which is the run viewer.
"""

from .model import FigureSpec, PanelRef, available_experiments, render
from .sources import PanelSource

__all__ = ["FigureSpec", "PanelRef", "PanelSource", "available_experiments", "render", "FigureWorkspace"]


def __getattr__(name: str):
    # Lazy so the model can be imported without a QApplication.
    if name == "FigureWorkspace":
        from .view import FigureWorkspace

        return FigureWorkspace
    raise AttributeError(name)

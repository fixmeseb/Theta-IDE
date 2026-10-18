"""Where panels come from: the plotters in plot/.

Two tiers, because not every plotter can draw into an Axes someone else
owns. Those that can (BasePlotter.panels()) become composable panels. Those
that build their own figures are offered through plot.capture instead, as
whole figures that can be restyled and exported but not merged into a grid.

Discovery is cached per (plotter, experiment): panels() reloads every
metrics.csv for the run, which is far too slow to repeat on each keystroke
in the styling controls.
"""

from __future__ import annotations

from typing import Any

from .model import PanelRef


class PanelSource:
    """The catalogue of panels available for an experiment."""

    def __init__(self) -> None:
        self._registry: dict[str, Any] | None = None
        self._panels: dict[tuple[str, str], list] = {}
        self._errors: dict[tuple[str, str], str] = {}

    # ── Plotters ─────────────────────────────────────────────────────────

    def registry(self) -> dict[str, Any]:
        if self._registry is None:
            from plot.manager import discover_plotters

            self._registry = discover_plotters()
        return self._registry

    def plotter_names(self) -> list[str]:
        return sorted(self.registry())

    # ── Composable panels ────────────────────────────────────────────────

    def _panels_for(self, plotter_name: str, exp_id: str) -> list:
        cache_key = (plotter_name, exp_id)
        if cache_key in self._panels:
            return self._panels[cache_key]

        cls = self.registry().get(plotter_name)
        if cls is None:
            self._panels[cache_key] = []
            return []
        try:
            found = cls().panels(exp_id)
        except Exception as exc:
            # One broken plotter must not empty the whole catalogue.
            self._errors[cache_key] = f"{type(exc).__name__}: {exc}"
            found = []
        self._panels[cache_key] = found
        return found

    def catalog(self, exp_id: str) -> list[PanelRef]:
        """Every panel that can be composed for *exp_id*."""
        refs: list[PanelRef] = []
        for plotter_name in self.plotter_names():
            for spec in self._panels_for(plotter_name, exp_id):
                refs.append(
                    PanelRef(key=spec.key, exp_id=exp_id, plotter=plotter_name, label=f"{plotter_name}: {spec.label}")
                )
        return refs

    def resolve(self, ref: PanelRef):
        """The PanelSpec a PanelRef names, or None when it is gone."""
        for spec in self._panels_for(ref.plotter, ref.exp_id):
            if spec.key == ref.key:
                return spec
        return None

    # ── Whole-figure fallback ────────────────────────────────────────────

    def capture(self, plotter_name: str, exp_id: str) -> list:
        """Figures from a plotter that cannot draw into a supplied Axes.

        Nothing is written to disk; see plot/capture.py.
        """
        from plot.capture import figures_from

        cls = self.registry().get(plotter_name)
        if cls is None:
            return []
        try:
            return figures_from(cls(), exp_id)
        except Exception as exc:
            self._errors[(plotter_name, exp_id)] = f"{type(exc).__name__}: {exc}"
            return []

    def composable(self, exp_id: str) -> list[str]:
        """Plotters offering panels for this experiment."""
        return [name for name in self.plotter_names() if self._panels_for(name, exp_id)]

    def errors(self) -> dict[tuple[str, str], str]:
        return dict(self._errors)

    def invalidate(self) -> None:
        """Drop the cache so a finished run shows its new metrics."""
        self._panels.clear()
        self._errors.clear()

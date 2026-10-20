"""Spec-driven sequential models, kept in the repo rather than installed.

src/usr/models/ is component territory, but this is not a research model: it
is the interpreter the visual model builder writes configs for. A button in
the Experiment panel cannot depend on a Hub install to work, so it stays
tracked through a .gitignore negation, the same way src/usr/methods/ keeps
agent_registry.py and base_agent.py while the rest of that directory is
component territory.

Importing the module is what registers the architecture; auto_discover_models
imports each package under src/usr/models/, so nothing in model_registry.py
had to change.
"""

from src.usr.models.sequential_spec import sequential_spec  # noqa: F401

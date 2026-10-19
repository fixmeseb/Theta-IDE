"""Visual model builder: design a network by wiring layer blocks together.

graph.py holds the rules and the saved format and imports no Qt, so it runs
headless in CI; canvas.py and dialog.py are the editor over it.
"""

from .graph import ModelGraph, model_config_yaml

__all__ = ["ModelGraph", "model_config_yaml", "ModelBuilderDialog"]


def __getattr__(name: str):
    # Imported lazily so `from frontend.model_builder import ModelGraph` stays
    # usable without a QApplication.
    if name == "ModelBuilderDialog":
        from .dialog import ModelBuilderDialog

        return ModelBuilderDialog
    raise AttributeError(name)

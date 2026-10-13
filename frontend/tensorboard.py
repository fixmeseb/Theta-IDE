"""TensorBoard facade maintaining backward compatibility for existing imports and tests."""
from frontend.plugins.core.tensorboard.server import TensorBoardServer, _free_port, READY_LINE
from frontend.plugins.core.tensorboard.panel import TensorBoardPanel

__all__ = ["TensorBoardServer", "TensorBoardPanel", "_free_port", "READY_LINE"]

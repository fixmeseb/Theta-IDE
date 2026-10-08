"""Engine and execution entrypoint resolution for NeSyRL pipeline.

Dispatches training to the appropriate script (defaults to src/app/train.py)
and handles framework engines (pytorch/lightning, sb3, cleanrl, custom scripts).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

DEFAULT_ENGINE_SCRIPT = "src/app/train.py"

ENGINE_SCRIPT_MAP = {
    "pytorch": "src/app/train.py",
    "lightning": "src/app/train.py",
    "sb3": "src/usr/methods/sb3_runner.py",
    "stable_baselines3": "src/usr/methods/sb3_runner.py",
    "cleanrl": "src/usr/methods/cleanrl_runner.py",
}


def resolve_engine(
    method_cfg: Optional[Dict[str, Any]] = None,
    cfg: Optional[Any] = None,
) -> Tuple[str, Optional[str]]:
    """Resolve the target script entrypoint and optional custom Python interpreter.

    Resolution order for script:
      1. method_cfg['entrypoint'] or method_cfg['runner'] or method_cfg['script']
      2. cfg.agent.entrypoint or cfg.agent.runner (if present)
      3. method_cfg['engine'] or cfg.agent.engine -> mapped via ENGINE_SCRIPT_MAP
      4. Fallback: "src/app/train.py"

    Resolution order for python interpreter:
      1. method_cfg['python'] or method_cfg['venv_python']
      2. method_cfg['venv'] -> resolves to <venv>/bin/python3
      3. Fallback: None (caller uses default project venv/python)
    """
    script = None
    python_bin = None

    m_cfg = method_cfg or {}

    # Check direct script declaration
    for key in ("entrypoint", "runner", "script"):
        val = m_cfg.get(key)
        if val:
            script = str(val)
            break

    # Check cfg.agent if available
    if not script and cfg is not None and hasattr(cfg, "agent") and cfg.agent is not None:
        agent_cfg = cfg.agent
        for key in ("entrypoint", "runner", "script"):
            val = getattr(agent_cfg, key, None) if not hasattr(agent_cfg, "get") else agent_cfg.get(key)
            if val:
                script = str(val)
                break

    # Check engine name (e.g. 'sb3', 'cleanrl', 'pytorch')
    if not script:
        engine_name = m_cfg.get("engine")
        if not engine_name and cfg is not None and hasattr(cfg, "agent") and cfg.agent is not None:
            engine_name = (
                getattr(cfg.agent, "engine", None)
                if not hasattr(cfg.agent, "get")
                else cfg.agent.get("engine")
            )

        if engine_name:
            clean_engine = str(engine_name).lower().strip()
            script = ENGINE_SCRIPT_MAP.get(clean_engine, clean_engine)

    # Fallback to standard PyTorch Lightning engine
    if not script:
        script = DEFAULT_ENGINE_SCRIPT

    # Check python interpreter overrides
    if "python" in m_cfg:
        python_bin = str(m_cfg["python"])
    elif "venv_python" in m_cfg:
        python_bin = str(m_cfg["venv_python"])
    elif "venv" in m_cfg:
        venv_dir = Path(m_cfg["venv"])
        candidate = venv_dir / "bin" / "python3"
        if candidate.exists():
            python_bin = str(candidate)

    return script, python_bin

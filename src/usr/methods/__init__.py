"""Legacy backward-compatibility forwarding module for src.usr.methods -> src.usr.algorithms."""

import sys

import src.usr.algorithms as algorithms
from src.usr.algorithms import agent_registry, base_agent, sb3_runner

# Re-export and register in sys.modules so direct subpackage imports continue to work
sys.modules.setdefault("src.usr.methods.agent_registry", agent_registry)
sys.modules.setdefault("src.usr.methods.base_agent", base_agent)
sys.modules.setdefault("src.usr.methods.sb3_runner", sb3_runner)

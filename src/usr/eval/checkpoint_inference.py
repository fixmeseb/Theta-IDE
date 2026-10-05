"""Inference helpers shared by evaluation and plotting: load an agent checkpoint and query its policy.

Used by src/usr/eval/pyrenees_evaluator.py (PyreneesEvaluator) and plot/clinical_alignment.py
(ClinicalAlignmentPlotter), which previously each kept their own copy.
"""

import torch


def load_agent(path, device, log_prefix="checkpoint_inference"):
    """Load an offline agent (CEW, CQL or IQL) from a Lightning checkpoint onto `device`, in eval mode.

    Tries each agent class, first strictly and then with strict=False; returns None (after printing the
    last error, prefixed with `log_prefix`) if none can load it.
    """
    from src.usr.methods.cew_agent import CEWAgent
    from src.usr.methods.cql_agent import CQLAgent
    from src.usr.methods.iql_agent import IQLAgent

    last_error = None
    for cls in [CQLAgent, CEWAgent, IQLAgent]:
        try:
            ag = cls.load_from_checkpoint(str(path), map_location=device, weights_only=False)
            ag.to(device)
            ag.eval()
            return ag
        except Exception as e:
            last_error = e
            try:
                ag = cls.load_from_checkpoint(str(path), map_location=device, weights_only=False, strict=False)
                ag.to(device)
                ag.eval()
                return ag
            except Exception as e2:
                last_error = e2
                continue
    if last_error is not None:
        print(f"  [{log_prefix}] Checkpoint load error for {path}: {last_error}")
    return None


def get_probs_and_actions(ag, obs_b):
    """Action probabilities and greedy actions of a loaded agent for a batch of observations."""
    if hasattr(ag, "get_action_probs"):
        probs = ag.get_action_probs(obs_b)
        acts = ag.get_action(obs_b) if hasattr(ag, "get_action") else torch.argmax(probs, dim=-1)
        return probs, acts

    # Fallback for unmigrated or raw models
    is_cql = ag.__class__.__name__ == "CQLAgent" or "cql" in str(getattr(ag, "algorithm", "")).lower()
    use_actor = bool(ag.get_cfg("use_actor", False)) if hasattr(ag, "get_cfg") else getattr(ag, "use_actor", False)

    if hasattr(ag, "is_modular") and ag.is_modular:
        logic_obs = (
            ag._prepare_logic_obs(obs_b)
            if hasattr(ag, "_prepare_logic_obs")
            else obs_b.unsqueeze(1).repeat(1, 2, 1)
        )
        if is_cql and not use_actor and hasattr(ag.model, "get_q_values"):
            q_vals = ag.model.get_q_values(obs_b, logic_obs)
            probs = torch.softmax(q_vals, dim=-1)
            acts = torch.argmax(q_vals, dim=-1)
            return probs, acts
        elif use_actor and hasattr(ag.model, "actor"):
            probs, _ = ag.model.actor(obs_b, logic_obs)
            acts = torch.argmax(probs, dim=-1)
            return probs, acts
        elif hasattr(ag.model, "get_q_values"):
            q_vals = ag.model.get_q_values(obs_b, logic_obs)
            probs = torch.softmax(q_vals, dim=-1)
            acts = torch.argmax(q_vals, dim=-1)
            return probs, acts
    elif is_cql and not use_actor and hasattr(ag, "q_network"):
        q = ag.q_network.get_q_values(obs_b) if hasattr(ag.q_network, "get_q_values") else ag.q_network(obs_b)
        probs = torch.softmax(q, dim=-1)
        acts = torch.argmax(probs, dim=-1)
        return probs, acts
    elif hasattr(ag, "actor") and hasattr(ag.actor, "get_action_probs"):
        probs = ag.actor.get_action_probs(obs_b)
        acts = torch.argmax(probs, dim=-1)
        return probs, acts
    elif hasattr(ag, "fuzzy_model") and ag.fuzzy_model is not None:
        q = ag.fuzzy_model(obs_b.to("cpu"))
        probs = torch.softmax(q, dim=-1).to(obs_b.device)
        acts = torch.argmax(probs, dim=-1)
        return probs, acts
    elif hasattr(ag, "q_network"):
        if hasattr(ag.q_network, "get_action_probs"):
            probs = ag.q_network.get_action_probs(obs_b)
        else:
            q = ag.q_network(obs_b)
            probs = torch.softmax(q, dim=-1)
        acts = torch.argmax(probs, dim=-1)
        return probs, acts
    elif hasattr(ag, "model") and hasattr(ag.model, "get_q_values"):
        q = ag.model.get_q_values(obs_b)
        probs = torch.softmax(q, dim=-1)
        acts = torch.argmax(probs, dim=-1)
        return probs, acts
    else:
        out = ag.get_action_and_value(obs_b)
        act = out[0] if isinstance(out, (tuple, list)) else out
        n_acts = 3 if obs_b.shape[-1] >= 123 else 2
        probs = torch.zeros((obs_b.shape[0], n_acts), device=obs_b.device)
        probs.scatter_(1, act.unsqueeze(1).long(), 1.0)
        return probs, act

"""Compatibility exports for the notebook-era ``src`` package."""

from ppo.actions import ACTION_NAMES, apply_action

ACTIONS = ACTION_NAMES

__all__ = ["ACTIONS", "ACTION_NAMES", "apply_action"]

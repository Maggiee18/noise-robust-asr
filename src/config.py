"""
Configuration loading.

    cfg = load_config("configs/model_b.yaml")

An experiment file may name a parent with `base: base.yaml`; the parent is
loaded first and the experiment's values are merged on top (nested keys are
merged, everything else is replaced). The result is a plain dict.
"""

import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _merge(base, over):
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    parent = cfg.pop("base", None)
    if parent:
        cfg = _merge(load_config(path.parent / parent), cfg)
    return cfg


def resolve(cfg, key):
    """Absolute path for a cfg['paths'] entry."""
    return ROOT / cfg["paths"][key]

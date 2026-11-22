"""Per-user settings layered over the shipped config.yaml.

config.yaml carries the defaults everyone starts from. A person's own choices
(full permissions for an assistant, tone of voice on, a different model) go in
~/.config/mcbeak/settings.yaml ($MCBEAK_SETTINGS overrides the path), so a
pull of new defaults never undoes them. Mappings merge key by key; any other
value in the settings file replaces the default.
"""
import os
from pathlib import Path
from typing import Optional

import yaml


def settings_path() -> Path:
    if os.environ.get("MCBEAK_SETTINGS"):
        return Path(os.path.expanduser(os.environ["MCBEAK_SETTINGS"]))
    base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return base / "mcbeak" / "settings.yaml"


def merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge(out[key], value)
        else:
            out[key] = value
    return out


def apply(config: dict, path: Optional[Path] = None) -> dict:
    path = path or settings_path()
    if not path.exists():
        return config
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as err:
        raise ValueError(f"{path} is not valid YAML: {err}")
    if not isinstance(data, dict):
        raise ValueError(f"{path} must hold settings as a mapping, like config.yaml")
    return merge(config, data)

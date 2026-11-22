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


def parse(text: str, source: str = "The settings text") -> dict:
    """Settings text as a mapping. Raises ValueError saying what is wrong with it."""
    try:
        data = yaml.safe_load(text) or {}
    except yaml.YAMLError as err:
        raise ValueError(f"{source} is not valid YAML: {err}")
    if not isinstance(data, dict):
        raise ValueError(f"{source} must hold settings as a mapping, like config.yaml")
    return data


def read_text(path: Optional[Path] = None) -> str:
    """The settings file as written, or "" when there is none."""
    path = path or settings_path()
    return path.read_text(encoding="utf-8") if path.exists() else ""


def apply(config: dict, path: Optional[Path] = None) -> dict:
    path = path or settings_path()
    if not path.exists():
        return config
    return merge(config, parse(path.read_text(encoding="utf-8"), str(path)))


def write_text(text: str, path: Optional[Path] = None) -> dict:
    """Replace the settings file with `text`, kept as written so its comments
    survive. Text that is not a YAML mapping is refused and the file is left
    alone. Returns the settings it holds."""
    data = parse(text)
    path = path or settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(text if text.endswith("\n") or not text else text + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return data

"""
backend/utils/config_loader.py

Centralized, cached loader for backend/config/health_rules.json.

Every engine in PACKS reads its thresholds, weights and reference data
from this single loader instead of hardcoding values. This keeps the
system config-driven and lets ops/nutrition teams tune rules without
touching code.
"""

from __future__ import annotations

import json
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict


class ConfigLoadError(RuntimeError):
    """Raised when health_rules.json cannot be located, read or parsed."""


class HealthRulesConfig:
    """
    Thread-safe, lazily-loaded wrapper around health_rules.json.

    The file is read once per process (unless `reload()` is called
    explicitly, e.g. from an admin endpoint after editing the rules)
    and cached in memory for fast repeated access by all engines.
    """

    _DEFAULT_CONFIG_PATH = (
        Path(__file__).resolve().parent.parent / "config" / "health_rules.json"
    )

    def __init__(self, config_path: Path | str | None = None) -> None:
        self._config_path = Path(config_path) if config_path else self._DEFAULT_CONFIG_PATH
        self._lock = threading.Lock()
        self._data: Dict[str, Any] | None = None

    def _load_from_disk(self) -> Dict[str, Any]:
        if not self._config_path.exists():
            raise ConfigLoadError(
                f"health_rules.json not found at expected path: {self._config_path}"
            )
        try:
            with self._config_path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except json.JSONDecodeError as exc:
            raise ConfigLoadError(
                f"health_rules.json is not valid JSON: {exc}"
            ) from exc

    def reload(self) -> None:
        """Force a re-read of the config file from disk."""
        with self._lock:
            self._data = self._load_from_disk()

    @property
    def data(self) -> Dict[str, Any]:
        """Return the full parsed configuration, loading it on first use."""
        if self._data is None:
            with self._lock:
                if self._data is None:  # double-checked locking
                    self._data = self._load_from_disk()
        return self._data

    def get(self, *keys: str, default: Any = None) -> Any:
        """
        Safely walk a nested path of keys inside the config.

        Example:
            config.get("nutrition_thresholds", "sugar", "high", default=22.5)
        """
        node: Any = self.data
        for key in keys:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    def section(self, name: str) -> Dict[str, Any]:
        """Return an entire top-level section (e.g. 'nutrition_thresholds')."""
        section = self.data.get(name)
        if section is None:
            raise ConfigLoadError(f"Section '{name}' missing from health_rules.json")
        return section


@lru_cache(maxsize=1)
def get_health_rules_config() -> HealthRulesConfig:
    """
    Process-wide singleton accessor.

    Using lru_cache(maxsize=1) instead of a bare module-level global keeps
    this test-friendly: `get_health_rules_config.cache_clear()` resets it.
    """
    return HealthRulesConfig()

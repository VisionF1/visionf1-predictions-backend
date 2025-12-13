"""
Caching utilities for prediction results.
"""

import json
import os
import re
import unicodedata
from typing import Any, Dict, Optional

def slugify(text: str) -> str:
    """Converts text to a safe filename slug."""
    norm = unicodedata.normalize("NFKD", text)
    norm = norm.encode("ascii", "ignore").decode("ascii")
    norm = norm.lower()
    norm = re.sub(r"[^a-z0-9]+", "_", norm)
    norm = norm.strip("_")
    return norm or "unknown"


def get_cache_path(base_dir: str, race_name: str, scenario: str) -> str:
    """Generates the file path for the cache entry."""
    race_slug = slugify(race_name)
    scen_slug = slugify(scenario)
    filename = f"{race_slug}__{scen_slug}.json"
    return os.path.join(base_dir, filename)


def load_cache(base_dir: str, race_name: str, scenario: str) -> Optional[Dict[str, Any]]:
    """Loads prediction result from JSON cache file."""
    path = get_cache_path(base_dir, race_name, scenario)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save_cache(base_dir: str, race_name: str, scenario: str, payload: Dict[str, Any]) -> None:
    """Saves prediction result to JSON cache file."""
    path = get_cache_path(base_dir, race_name, scenario)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
    except Exception:
        pass

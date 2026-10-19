"""Per-user settings saved as JSON files (user_data/<username>.json)."""
import json
import os
import re
from pathlib import Path

DIR = Path(os.environ.get("BACKTEST_DATA_DIR", "user_data"))


def _path(user: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", user) or "user"
    return DIR / f"{safe}.json"


def load(user: str) -> dict:
    try:
        return json.loads(_path(user).read_text(encoding="utf-8"))
    except Exception:
        return {}


def save(user: str, cfg: dict) -> bool:
    try:
        DIR.mkdir(parents=True, exist_ok=True)
        p = _path(user)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(cfg, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        os.replace(tmp, p)  # atomic
        return True
    except Exception:
        return False

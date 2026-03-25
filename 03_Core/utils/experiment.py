from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, is_dataclass
from typing import Any, Mapping


def make_run_dir(base_dir: str, run_name: str | None = None) -> str:
    os.makedirs(base_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    name = f"{stamp}_{run_name}" if run_name else stamp
    run_dir = os.path.join(base_dir, name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def save_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def save_config(run_dir: str, args: Any, extra: Mapping[str, Any] | None = None) -> None:
    if hasattr(args, "__dict__"):
        base = dict(args.__dict__)
    elif is_dataclass(args):
        base = asdict(args)
    else:
        base = dict(args)

    if extra:
        base.update(dict(extra))

    save_json(os.path.join(run_dir, "config.json"), base)

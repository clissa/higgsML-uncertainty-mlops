"""Local run index helpers.

Maintains a human-readable, append-only JSON list at
``results/runs_index.json`` (or a configured path) that every run
appends to — enabling offline run discovery and comparison.

Usage::

    from conformal_predictions.mlops.run_index import append_run, load_index

    append_run(record, "results/runs_index.json")

    past_runs = load_index("results/runs_index.json")
    for r in past_runs:
        print(r["run_id"], r["timestamp"], r["dataset"])
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


def append_run(record: dict, index_path: "str | Path") -> None:
    """Append a run summary record to the index file.

    The write is atomic (temp-file + ``os.replace``) to be safe against
    partial writes when multiple processes run on the same machine.

    Parameters
    ----------
    record : dict
        Run summary dict (at minimum: ``run_id``, ``timestamp``,
        ``dataset``, ``output_dir``, ``git_commit``, ``metrics``).
    index_path : str | Path
        Path to the JSON index file.  The parent directory is created
        automatically if it does not exist.
    """
    index_path = Path(index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)

    # Read existing records
    if index_path.exists():
        try:
            with open(index_path) as fh:
                existing: list = json.load(fh)
        except (json.JSONDecodeError, OSError):
            existing = []
    else:
        existing = []

    existing.append(record)

    # Atomic write via temp-file + os.replace
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=index_path.parent,
        prefix=".runs_index_",
        suffix=".json",
    )
    try:
        with os.fdopen(tmp_fd, "w") as fh:
            json.dump(existing, fh, indent=2)
        os.replace(tmp_path, index_path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def load_index(index_path: "str | Path") -> list:
    """Return the list of run records from the index file.

    Parameters
    ----------
    index_path : str | Path
        Path to the JSON index file.

    Returns
    -------
    list
        List of run record dicts.  Returns an empty list if the file
        does not exist or cannot be parsed.
    """
    index_path = Path(index_path)
    if not index_path.exists():
        return []
    try:
        with open(index_path) as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return []

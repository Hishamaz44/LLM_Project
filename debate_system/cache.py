"""Disk-backed cache for LLM calls, so repeated/debugging runs never re-pay for a call already made."""

import hashlib
import json
import os
import tempfile
from pathlib import Path

DEFAULT_CACHE_PATH = Path(__file__).resolve().parent.parent / "cache.json"


# Hashes the call parameters into a stable lookup key for the cache.
def make_key(model: str, prompt: str, max_tokens: int, temperature: float, slot: int) -> str:
    raw = f"{model}||{prompt}||{max_tokens}||{temperature}||{slot}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# Simple key/value cache backed by a single JSON file on disk.
class Cache:
    # Constructor: loads existing cache contents from disk, or starts empty.
    def __init__(self, path: Path = DEFAULT_CACHE_PATH):
        self.path = Path(path)
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text())
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Cache file {self.path} is corrupt and could not be parsed ({exc}). "
                    f"Delete it to start fresh (you will re-pay for calls made so far)."
                ) from exc
        else:
            self._data = {}

    # Getter: returns the cached value for `key`, or None if not cached.
    def get(self, key: str) -> str | None:
        return self._data.get(key)

    # Setter: stores `value` under `key` and persists the whole cache to disk.
    def set(self, key: str, value: str) -> None:
        self._data[key] = value
        self._atomic_write()

    # Writes the cache to a temp file in the same directory, then atomically renames it
    # over the real file — so an interrupted write can never leave a half-written cache.
    def _atomic_write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(self._data, f)
            os.replace(tmp_path, self.path)
        except BaseException:
            Path(tmp_path).unlink(missing_ok=True)
            raise

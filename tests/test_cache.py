import json

import pytest

from debate_system.cache import Cache, make_key


def test_cache_is_a_miss_then_a_hit_after_set(tmp_path):
    cache = Cache(path=tmp_path / "cache.json")
    key = make_key("model-a", "prompt", 100, 0.0, 0)
    assert cache.get(key) is None
    cache.set(key, "response text")
    assert cache.get(key) == "response text"


def test_cache_persists_to_disk_across_instances(tmp_path):
    path = tmp_path / "cache.json"
    key = make_key("model-a", "prompt", 100, 0.0, 0)

    cache1 = Cache(path=path)
    cache1.set(key, "response text")

    cache2 = Cache(path=path)
    assert cache2.get(key) == "response text"


def test_make_key_differs_by_slot_so_homogeneous_judges_are_independent():
    key0 = make_key("model-a", "prompt", 100, 0.7, slot=0)
    key1 = make_key("model-a", "prompt", 100, 0.7, slot=1)
    assert key0 != key1


def test_make_key_differs_by_temperature_and_model():
    base = make_key("model-a", "prompt", 100, 0.0, 0)
    assert base != make_key("model-a", "prompt", 100, 0.7, 0)
    assert base != make_key("model-b", "prompt", 100, 0.0, 0)


def test_cache_set_writes_valid_json_and_leaves_no_temp_files(tmp_path):
    path = tmp_path / "cache.json"
    cache = Cache(path=path)
    cache.set(make_key("m", "p", 100, 0.0, 0), "value")

    # File is valid JSON (an interrupted write would have left it truncated instead).
    assert json.loads(path.read_text()) != {}
    # No leftover temp files from the atomic write.
    assert list(tmp_path.glob("*.tmp")) == []


def test_corrupt_cache_file_raises_a_clear_error(tmp_path):
    path = tmp_path / "cache.json"
    path.write_text("{ this is not valid json")
    with pytest.raises(RuntimeError, match="corrupt"):
        Cache(path=path)

import pytest

from debate_system import client
from debate_system.cache import Cache, make_key


@pytest.fixture
def isolated_cache(tmp_path, monkeypatch):
    """Point the client at a throwaway cache so tests never touch the real cache.json."""
    cache = Cache(path=tmp_path / "cache.json")
    monkeypatch.setattr(client, "_cache", cache)
    return cache


def test_call_model_retries_past_a_bad_response_and_caches_only_the_good_one(
    isolated_cache, monkeypatch
):
    responses = iter(["bad", "good"])
    calls = []

    def fake_raw_call(model, prompt, max_tokens, temperature):
        calls.append(model)
        return next(responses)

    monkeypatch.setattr(client, "_raw_call", fake_raw_call)

    result = client.call_model("m", "prompt", validate=lambda text: text == "good")

    assert result == "good"
    assert len(calls) == 2  # retried once past the bad response
    key = make_key("m", "prompt", 400, 0.0, 0)
    assert isolated_cache.get(key) == "good"  # only the good response was cached


def test_call_model_serves_a_cached_response_without_hitting_the_api(
    isolated_cache, monkeypatch
):
    key = make_key("m", "prompt", 400, 0.0, 0)
    isolated_cache.set(key, "already cached")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("_raw_call should not run on a cache hit")

    monkeypatch.setattr(client, "_raw_call", fail_if_called)

    assert client.call_model("m", "prompt") == "already cached"


def test_call_model_raises_after_exhausting_attempts_on_persistent_bad_output(
    isolated_cache, monkeypatch
):
    monkeypatch.setattr(client, "_raw_call", lambda *a, **k: "bad")

    with pytest.raises(RuntimeError, match="after 3 attempts"):
        client.call_model("m", "prompt", validate=lambda text: False)

    assert isolated_cache.get(make_key("m", "prompt", 400, 0.0, 0)) is None

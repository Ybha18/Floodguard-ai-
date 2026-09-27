import types

from floodguard.orchestrator import _extract_json_object, copilot
from floodguard.providers import _strip_reasoning


def test_strip_reasoning_removes_closed_think_block():
    raw = "<think>long internal reasoning trace here</think>Route index 1 is recommended."
    assert _strip_reasoning(raw) == "Route index 1 is recommended."


def test_strip_reasoning_removes_unclosed_think_block():
    raw = "<think>Analyze user input... still thinking and never finishes"
    assert _strip_reasoning(raw) == ""


def test_strip_reasoning_leaves_normal_text_untouched():
    raw = "Route index 1 has the lowest flood exposure at 13%."
    assert _strip_reasoning(raw) == raw


def test_extract_json_object_from_plain_json():
    text = '{"agreement":"CONSENSUS","answer":"Take route 1."}'
    assert _extract_json_object(text) == {"agreement": "CONSENSUS", "answer": "Take route 1."}


def test_extract_json_object_from_fenced_json():
    text = '```json\n{"agreement":"DISAGREEMENT","answer":"Prefer deterministic data."}\n```'
    parsed = _extract_json_object(text)
    assert parsed["agreement"] == "DISAGREEMENT"


def test_extract_json_object_from_json_with_surrounding_prose():
    text = 'Here is the result: {"agreement":"PARTIAL_AGREEMENT","answer":"Route 1 is less exposed."} End.'
    assert _extract_json_object(text)["answer"] == "Route 1 is less exposed."


def test_extract_json_object_returns_none_on_garbage():
    assert _extract_json_object("not json at all") is None


def test_provider_empty_visible_content_does_not_fallback_to_reasoning(monkeypatch):
    from floodguard import providers

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "reasoning_content": "<think>I know the answer</think> but it should not leak",
                        }
                    }
                ]
            }

    captured = {}

    def fake_post(*args, **kwargs):
        captured.update(kwargs["json"])
        return Response()

    monkeypatch.setattr(providers.requests, "post", fake_post)
    text, _ = providers._post_openai_compatible(
        "https://example.test/v1",
        "key",
        "nvidia/nemotron-3-super-120b-a12b",
        "hello",
        enable_thinking=True,
        max_tokens=16384,
    )
    assert text == ""
    assert captured["chat_template_kwargs"] == {"enable_thinking": True}
    assert captured["max_tokens"] == 16384


def test_primary_nemotron_payload_uses_real_thinking_control(monkeypatch):
    from floodguard import providers

    captured = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "PRIMARY OK"}}]}

    def fake_post(*args, **kwargs):
        captured.update(kwargs["json"])
        return Response()

    fake_settings = types.SimpleNamespace(
        nvidia_api_key="nvidia-key",
        nvidia_primary_model="nvidia/nemotron-3.5-lightning-30b-a3b",
        nvidia_synthesis_model="nvidia/nemotron-3-super-120b-a12b",
    )
    monkeypatch.setattr(providers, "SETTINGS", fake_settings)
    monkeypatch.setattr(providers.requests, "post", fake_post)

    text, _ = providers.call_nemotron("analyze", role="primary")
    assert text == "PRIMARY OK"
    assert captured["model"] == "nvidia/nemotron-3.5-lightning-30b-a3b"
    assert captured["chat_template_kwargs"] == {"enable_thinking": True}
    assert captured["max_tokens"] >= 8192
    assert captured["thinking_token_budget"] < captured["max_tokens"]


def test_synthesis_nemotron_payload_uses_super_and_reasoning(monkeypatch):
    from floodguard import providers

    captured = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": '{"agreement":"CONSENSUS","answer":"READY"}'}}]}

    def fake_post(*args, **kwargs):
        captured.update(kwargs["json"])
        return Response()

    fake_settings = types.SimpleNamespace(
        nvidia_api_key="nvidia-key",
        nvidia_primary_model="nvidia/nemotron-3.5-lightning-30b-a3b",
        nvidia_synthesis_model="nvidia/nemotron-3-super-120b-a12b",
    )
    monkeypatch.setattr(providers, "SETTINGS", fake_settings)
    monkeypatch.setattr(providers.requests, "post", fake_post)

    text, _ = providers.call_nemotron("synthesize", role="synthesis")
    assert text.startswith('{"agreement"')
    assert captured["model"] == "nvidia/nemotron-3-super-120b-a12b"
    assert captured["chat_template_kwargs"] == {"enable_thinking": True}
    assert captured["max_tokens"] == 16384


def test_copilot_runs_analysis_second_opinion_and_final_synthesis(monkeypatch):
    from floodguard import orchestrator

    calls = []

    def fake_nemotron(prompt, role="primary"):
        calls.append(("Nemotron", role, prompt))
        if role == "primary":
            return "Primary says route 1 has lower simulated exposure.", 80.0
        return '{"agreement":"CONSENSUS","answer":"Use route 1 based on the verified route evidence."}', 140.0

    def fake_gemini(prompt):
        calls.append(("Gemini", "second", prompt))
        return '{"agreement":"CONSENSUS","analysis":"The primary analysis matches the supplied route evidence."}', 90.0

    fake_settings = types.SimpleNamespace(
        enable_gemini=True,
        gemini_api_key="gemini-key",
        enable_muse=False,
        nvidia_api_key="nvidia-key",
    )
    monkeypatch.setattr(orchestrator, "SETTINGS", fake_settings)
    monkeypatch.setattr(orchestrator, "call_nemotron", fake_nemotron)
    monkeypatch.setattr(orchestrator, "call_gemini", fake_gemini)

    result = copilot("Which route should I take?", {"run_id": "test", "routes": [1]})
    assert result["agreement"] == "CONSENSUS"
    assert result["answer"] == "Use route 1 based on the verified route evidence."
    assert "Nemotron Primary" in result["responses"]
    assert "Gemini Second Opinion" in result["responses"]
    assert "Nemotron Super" in result["responses"]
    assert result["latencies"]["Nemotron Super"] == 140.0
    assert any(role == "primary" for _, role, _ in calls)
    assert any(role == "synthesis" for _, role, _ in calls)


def test_copilot_gemini_is_optional_and_final_super_can_continue(monkeypatch):
    from floodguard import orchestrator

    def fake_nemotron(prompt, role="primary"):
        if role == "primary":
            return "Primary: current run shows route 2 has lower exposure.", 70.0
        return '{"agreement":"PARTIAL_AGREEMENT","answer":"The evidence supports route 2, with prototype-model uncertainty."}', 130.0

    fake_settings = types.SimpleNamespace(
        enable_gemini=False,
        gemini_api_key=None,
        enable_muse=False,
        nvidia_api_key="nvidia-key",
    )
    monkeypatch.setattr(orchestrator, "SETTINGS", fake_settings)
    monkeypatch.setattr(orchestrator, "call_nemotron", fake_nemotron)

    result = copilot("Why did the route change?", {"run_id": "test"})
    assert result["agreement"] == "PARTIAL_AGREEMENT"
    assert "route 2" in result["answer"]
    assert "Gemini Second Opinion" in result["errors"]


def test_copilot_no_provider_returns_no_model(monkeypatch):
    from floodguard import orchestrator

    def boom(*_args, **_kwargs):
        raise RuntimeError("API key not configured")

    fake_settings = types.SimpleNamespace(
        enable_gemini=True,
        gemini_api_key="gemini-key",
        enable_muse=True,
        nvidia_api_key="nvidia-key",
    )
    monkeypatch.setattr(orchestrator, "SETTINGS", fake_settings)
    monkeypatch.setattr(orchestrator, "call_nemotron", boom)
    monkeypatch.setattr(orchestrator, "call_gemini", boom)
    monkeypatch.setattr(orchestrator, "call_muse", boom)

    result = copilot("Which route should I take?", {"run_id": "test"})
    assert result["agreement"] == "NO MODEL"
    assert result["answer"].startswith("No enabled AI")

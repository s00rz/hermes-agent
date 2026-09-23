"""Behavioral coverage for generic final-text completion gates."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest, get_pre_completion_directive


def _response(content="draft answer"):
    message = SimpleNamespace(content=content, tool_calls=None)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="stop")],
        model="test/model",
        usage=None,
    )


@pytest.fixture
def agent(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setenv("HERMES_VERIFY_ON_STOP", "0")
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        instance = AIAgent(
            session_id="completion-test",
            api_key="test-key",
            base_url="https://example.invalid/v1",
            provider="openai-compat",
            model="test/model",
            max_iterations=1,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    instance._cached_system_prompt = "stable test prompt"
    instance._session_db = None
    instance.save_trajectories = False
    instance.compression_enabled = False
    instance._cleanup_task_resources = lambda *_a, **_kw: None
    instance._save_trajectory = lambda *_a, **_kw: None
    return instance


@pytest.mark.parametrize("scenario", ["continue_then_allow", "continue_at_budget", "fail", "malformed"])
def test_pre_completion_outcomes_preserve_completion_contract(agent, scenario):
    if scenario == "continue_then_allow":
        agent.max_iterations = 2
        agent.iteration_budget.max_total = 2
        answers = iter([_response("draft answer"), _response("repaired answer")])
        agent._interruptible_api_call = lambda _kwargs: next(answers)
    else:
        agent._interruptible_api_call = lambda _kwargs: _response()
    agent._handle_max_iterations = MagicMock(return_value="replacement summary")
    observed = []

    def invoke(hook_name, **kwargs):
        if hook_name == "pre_completion":
            observed.append(kwargs)
            if scenario in {"continue_then_allow", "continue_at_budget"} and len(observed) == 1:
                return [{"action": "continue", "message": "finish the required check"}]
            if scenario == "continue_at_budget":
                return [{"action": "continue", "message": "finish the required check"}]
            if scenario == "fail":
                return [{"action": "fail", "reason": "completion_check_unavailable"}]
            if scenario == "malformed":
                return [{"action": "continue"}]
        return []

    with patch("hermes_cli.plugins.invoke_hook", side_effect=invoke):
        result = agent.run_conversation("complete the task")

    assert [item["attempt"] for item in observed] == (
        [0, 1] if scenario == "continue_then_allow" else [0]
    )
    assert observed[0] == {
        "session_id": "completion-test",
        "platform": "",
        "attempt": 0,
        "final_response": "draft answer",
    }
    if scenario == "continue_then_allow":
        assert result["completed"] is True
        assert result["final_response"] == "repaired answer"
        assert result["turn_exit_reason"] == "text_response(finish_reason=stop)"
        assert [message["role"] for message in result["messages"]] == ["user", "assistant", "assistant"]
        assert result["messages"][1]["content"] == "draft answer"
        agent._handle_max_iterations.assert_not_called()
    elif scenario == "continue_at_budget":
        assert result["completed"] is False
        assert result["final_response"] == "draft answer"
        assert result["turn_exit_reason"] == "max_iterations_reached(1/1)"
        assert [message["role"] for message in result["messages"]] == ["user", "assistant"]
        assert not result["messages"][1].get("_pre_completion_synthetic")
        agent._handle_max_iterations.assert_not_called()
    else:
        expected_reason = (
            "completion_check_unavailable" if scenario == "fail"
            else "pre_completion continue directive was malformed"
        )
        assert result["completed"] is False
        assert result["failed"] is True
        assert result["final_response"] == f"Completion check failed: {expected_reason}"
        assert result["turn_exit_reason"] == f"pre_completion_rejected: {expected_reason}"
        assert result["failure_reason"] == "loop_error"
        assert result["messages"][-1]["role"] == "assistant"
        assert result["messages"][-1]["content"] == "draft answer"


def test_pre_completion_plugin_dispatch_is_profile_scoped_and_session_concurrent(tmp_path):
    from agent.shell_hooks import _parse_hooks_block

    assert _parse_hooks_block({"pre_completion": [{"command": "unused"}]}) == []

    release = Event()
    all_started = Event()
    lock = Lock()
    calls = []

    def make_manager(profile):
        manager = PluginManager(scope_key=str(tmp_path / profile))
        manifest = PluginManifest(name=f"completion-{profile}", key=f"completion-{profile}", source="user")
        ctx = PluginContext(manifest, manager)

        def callback(session_id, attempt, **_kwargs):
            with lock:
                calls.append((profile, session_id, attempt))
                if len(calls) >= 3:
                    all_started.set()
            assert release.wait(3)
            return {"action": "allow"}

        ctx.register_hook("pre_completion", callback)
        return manager, ctx

    first, first_ctx = make_manager("profile-a")
    second, second_ctx = make_manager("profile-b")

    def raises(**_kwargs):
        raise RuntimeError("callback failed")

    first_ctx.register_hook("pre_completion", raises)
    second_ctx.register_hook("pre_completion", lambda **_kwargs: {"action": "maybe"})

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [
            pool.submit(first.invoke_hook, "pre_completion", session_id="session-a", attempt=0),
            pool.submit(first.invoke_hook, "pre_completion", session_id="session-b", attempt=0),
            pool.submit(second.invoke_hook, "pre_completion", session_id="session-c", attempt=0),
        ]
        assert all_started.wait(2), "concurrent sessions or profile-scoped callbacks were suppressed"
        release.set()
        results = [future.result(timeout=3) for future in futures]

    assert set(calls) == {
        ("profile-a", "session-a", 0),
        ("profile-a", "session-b", 0),
        ("profile-b", "session-c", 0),
    }
    assert any(result.get("action") == "fail" for result in results[0])
    assert any(result.get("action") == "fail" for result in results[1])
    assert {result.get("action") for result in results[2]} == {"allow", "maybe"}

    with patch("hermes_cli.plugins.invoke_hook", second.invoke_hook):
        directive = get_pre_completion_directive(
            session_id="session-c", platform="cli", attempt=0, final_response="draft",
        )
    assert directive == {"action": "fail", "reason": "pre_completion hook returned an unknown action"}

    timeout_started = Event()
    release_timeout = Event()
    timeout_manager = PluginManager(scope_key=str(tmp_path / "profile-timeout"))
    timeout_manifest = PluginManifest(
        name="completion-timeout", key="completion-timeout", source="user",
    )
    timeout_ctx = PluginContext(timeout_manifest, timeout_manager)

    def hangs(**_kwargs):
        timeout_started.set()
        release_timeout.wait()

    timeout_ctx.register_hook("pre_completion", hangs)
    try:
        with ThreadPoolExecutor(max_workers=1) as pool, patch(
            "hermes_cli.plugins._resolve_hook_callback_timeout", return_value=2.1,
        ):
            future = pool.submit(
                timeout_manager.invoke_hook, "pre_completion", session_id="session-timeout", attempt=0,
            )
            assert timeout_started.wait(5), "bounded callback did not start"
            timeout_results = future.result(timeout=8)
        assert timeout_results == [{
            "action": "fail",
            "reason": "pre_completion hook callback timed out or is still running",
        }]
    finally:
        release_timeout.set()

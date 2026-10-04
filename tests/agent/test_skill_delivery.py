"""Instructional tool loads must reach the next model request whole, not as spill previews."""

import json
from types import SimpleNamespace

import pytest

from run_agent import AIAgent
from tools.budget_config import BudgetConfig
from tools.tool_result_storage import PERSISTED_OUTPUT_TAG, enforce_turn_budget, maybe_persist_tool_result


def _call(name, args, call_id):
    return SimpleNamespace(id=call_id, type="function", function=SimpleNamespace(
        name=name, arguments=json.dumps(args)))


@pytest.mark.parametrize("batch", [False, True])
def test_native_skill_delivery_preserves_full_instructions(tmp_path, monkeypatch, batch):
    home = tmp_path / "home"
    skill_dir = home / "skills" / "delivery-proof"
    skill_dir.mkdir(parents=True)
    # One-line JSON includes escaped newlines, quotes, non-ASCII and a long single line.
    content = '---\nname: delivery-proof\ndescription: Offline delivery proof\n---\n' + (
        'Read all instructions: "元宝 café 🧪"\n' * 4000) + "tail:" + "z" * 4000
    reference = 'Linked reference: "元宝 café 🧪"\n' * 4000 + "REFERENCE END"
    (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")
    (skill_dir / "references").mkdir()
    (skill_dir / "references" / "guide.md").write_text(reference, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.chdir(tmp_path)
    agent = AIAgent(model="offline-skill-proof", provider="openai",
        api_key="offline-not-a-credential", base_url="http://127.0.0.1:1/v1",
        enabled_toolsets=["skills"], quiet_mode=True, skip_context_files=True,
        skip_memory=True, save_trajectories=False)
    try:
        # The production executor derives the budget from this active context length.
        agent.context_compressor.context_length = 65536 if not batch else 200000
        calls = [_call("skill_view", {"name": "delivery-proof"}, "main")]
        expected = [content]
        if batch:
            calls.append(
                _call("skill_view", {"name": "delivery-proof", "file_path": "references/guide.md"}, "ref")
            )
            expected.append(reference)
        messages = []
        agent._execute_tool_calls(SimpleNamespace(content="", tool_calls=calls), messages, str(tmp_path))
        assert len(messages) == len(calls)
        for message, instructions in zip(messages, expected):
            assert message["name"] == "skill_view"
            assert PERSISTED_OUTPUT_TAG not in message["content"]
            payload = json.loads(message["content"])
            assert payload["success"] is True
            assert payload["content"] == instructions
        assert not list((home / "cache" / "spillover").glob("*.txt"))
    finally:
        agent.close()


def test_skill_exemption_does_not_remove_other_tool_budgets(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    budget = BudgetConfig(default_result_size=8000, turn_budget=16000,
        tool_overrides={"skill_view": 1})
    instructions = json.dumps({"success": True, "content": "instruction\n" * 10000})
    assert maybe_persist_tool_result(instructions, "skill_view", "skill", config=budget) == instructions
    assert maybe_persist_tool_result(instructions, "skill_view", "forced", config=budget, threshold=0) == instructions
    rows = [{"role": "tool", "tool_name": "skill_view", "content": instructions, "tool_call_id": "s"},
        {"role": "tool", "name": "terminal", "content": "ordinary" * 2000, "tool_call_id": "t"}]
    enforce_turn_budget(rows, config=budget)
    assert rows[0]["content"] == instructions
    assert PERSISTED_OUTPUT_TAG in rows[1]["content"]
    ordinary = maybe_persist_tool_result("ordinary" * 2000, "terminal", "single", config=budget)
    assert PERSISTED_OUTPUT_TAG in ordinary

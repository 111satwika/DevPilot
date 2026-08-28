"""Tests for ml/data/generate_adversarial.py -- the fine-tune's
mode-violation-refusal and explore-first data buckets, built from
DevPilot's real mode/tool configuration rather than a teacher API.
"""

import pytest

from llm.agent import GATED_TOOLS, PLAN_MODE_ALLOWED_TOOLS
from ml.data.generate_adversarial import generate_adversarial_examples


@pytest.mark.asyncio
async def test_generates_a_reasonable_volume_deterministically():
    examples_1 = await generate_adversarial_examples()
    examples_2 = await generate_adversarial_examples()
    assert len(examples_1) > 20
    # Same seed -> same output every run, so the dataset is reproducible.
    assert [e.user_request for e in examples_1] == [e.user_request for e in examples_2]


@pytest.mark.asyncio
async def test_ask_mode_refusals_have_zero_tools_offered():
    examples = await generate_adversarial_examples()
    ask_examples = [e for e in examples if e.mode == "ask"]
    assert ask_examples
    for ex in ask_examples:
        assert ex.tools == []
        assert ex.completion.type == "refusal"
        assert ex.completion.tool_calls == []


@pytest.mark.asyncio
async def test_plan_mode_refusals_only_cover_genuinely_excluded_tools():
    examples = await generate_adversarial_examples()
    plan_examples = [e for e in examples if e.mode == "plan" and e.source == "adversarial_mode_violation"]
    assert plan_examples
    for ex in plan_examples:
        assert ex.tool_family_holdout not in PLAN_MODE_ALLOWED_TOOLS
        assert ex.completion.type == "refusal"
        offered_names = {t["function"]["name"] for t in ex.tools}
        assert ex.tool_family_holdout not in offered_names


@pytest.mark.asyncio
async def test_every_gated_tool_has_at_least_one_refusal_example():
    """Catches silent drift -- a new gated tool added later without a
    matching adversarial template would otherwise go unnoticed."""
    examples = await generate_adversarial_examples()
    covered = {e.tool_family_holdout for e in examples if e.source == "adversarial_mode_violation"}
    assert GATED_TOOLS.issubset(covered)


@pytest.mark.asyncio
async def test_vague_requests_produce_list_directory_not_a_clarifying_question():
    """The important correction vs. the generic fine-tune plan: DevPilot's
    real SYSTEM_PROMPT (Entry 29) wants exploration, not a question."""
    examples = await generate_adversarial_examples()
    explore_examples = [e for e in examples if e.source == "adversarial_explore"]
    assert explore_examples
    for ex in explore_examples:
        assert ex.completion.type == "tool_call"
        assert ex.completion.tool_calls[0].name == "list_directory"


@pytest.mark.asyncio
async def test_system_prompt_is_the_real_one_by_default():
    from llm.agent import SYSTEM_PROMPT

    examples = await generate_adversarial_examples()
    assert examples[0].system_prompt == SYSTEM_PROMPT
    assert examples[0].system_prompt != ""


def test_read_only_refusal_templates_never_literally_match_positive_templates():
    """Entry 54's real regression: READ_ONLY_REQUEST_TEMPLATES (refusal,
    mode=ask) used to share exact request text with
    generate_positive_examples.py's POSITIVE_TEMPLATES (tool_call,
    mode=agent) for the same tool -- e.g. "what's in {filename}?" labeled
    refusal in one generator and tool_call in the other. A real Colab
    retrain on data containing that overlap produced a model that emitted
    Plan/Ask-style refusal text even for mode=agent requests. This must
    never silently come back."""
    from ml.data.generate_adversarial import READ_ONLY_REQUEST_TEMPLATES
    from ml.data.generate_positive_examples import POSITIVE_TEMPLATES

    checked = 0
    for tool_name, refusal_phrasings in READ_ONLY_REQUEST_TEMPLATES.items():
        positive_phrasings = {p for p, _ in POSITIVE_TEMPLATES.get(tool_name, [])}
        checked += 1
        assert not (set(refusal_phrasings) & positive_phrasings), (
            f"{tool_name}: refusal and tool_call templates share literal text"
        )
    assert checked >= 3  # sanity: read_file, git_log, list_tables all actually compared

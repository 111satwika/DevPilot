"""Tests for ml/data/generate_positive_examples.py -- the fix for
Entry 51's class-imbalance hypothesis (the dataset's only tool_call
examples all targeted list_directory).

The most important test here isn't about volume -- it's that the request
text and the labeled tool-call arguments actually agree with each other.
A first draft of this generator picked them from two independent
rng.choice() calls and could silently produce e.g. a request naming one
filename while the "correct" label pointed at a different one -- which
would train the model on subtly wrong data. That bug was caught by hand
before any test existed for it; this test exists so it can't come back
silently.
"""

import re

import pytest

from ml.data.generate_positive_examples import POSITIVE_TEMPLATES, generate_positive_examples

# Slot names that plausibly appear in both the request text and the
# argument dict, per tool -- used to check text/argument agreement
# without hardcoding every template's exact wording.
_TEXT_ARG_KEYS = {
    "read_file": "path", "get_file_info": "path", "git_diff": "path",
    "list_directory": "path",
    "describe_table": "table",
    "inspect_container": "container", "get_container_logs": "container",
    "get_pull_request": "number",
    "get_commit": "sha",
    "search_web": "query",
    "fetch_page": "url",
}


@pytest.mark.asyncio
async def test_generates_examples_across_many_tools_not_just_one():
    examples = await generate_positive_examples()
    tools_used = {e.tool_family_holdout for e in examples}
    assert len(tools_used) >= 15  # was 1 (list_directory only) before this generator existed


@pytest.mark.asyncio
async def test_every_template_tool_actually_produces_examples():
    examples = await generate_positive_examples()
    produced = {e.tool_family_holdout for e in examples}
    assert produced == set(POSITIVE_TEMPLATES)


@pytest.mark.asyncio
async def test_request_text_and_tool_call_arguments_agree():
    """The real bug this generator had: text and arguments picked
    independently, so they could name different values. For every tool
    where a value plausibly appears in both, the same literal value must
    appear in both the request text and the labeled argument."""
    examples = await generate_positive_examples()
    checked = 0
    for ex in examples:
        arg_key = _TEXT_ARG_KEYS.get(ex.tool_family_holdout)
        if arg_key is None:
            continue
        value = ex.completion.tool_calls[0].arguments.get(arg_key)
        if value is None:
            continue
        checked += 1
        assert str(value) in ex.user_request, (
            f"{ex.tool_family_holdout}: argument {arg_key}={value!r} not found in "
            f"request text {ex.user_request!r} -- text/argument mismatch"
        )
    assert checked > 10  # sanity: the check itself actually exercised something


@pytest.mark.asyncio
async def test_github_examples_use_the_same_owner_repo_in_text_and_arguments():
    """Same check as above, specifically for the two-part owner/repo
    case that originally crashed with a bare KeyError before either
    value existed as a fill slot at all."""
    examples = await generate_positive_examples()
    github_tools = {"get_repository", "list_pull_requests", "get_pull_request", "list_commits", "get_commit"}
    checked = 0
    for ex in examples:
        if ex.tool_family_holdout not in github_tools:
            continue
        args = ex.completion.tool_calls[0].arguments
        checked += 1
        assert args["owner"] in ex.user_request
        assert args["repo"] in ex.user_request
    assert checked >= 5


@pytest.mark.asyncio
async def test_is_deterministic_across_runs():
    examples_1 = await generate_positive_examples()
    examples_2 = await generate_positive_examples()
    assert [e.user_request for e in examples_1] == [e.user_request for e in examples_2]
    assert [e.completion.tool_calls[0].arguments for e in examples_1] == [
        e.completion.tool_calls[0].arguments for e in examples_2
    ]


@pytest.mark.asyncio
async def test_every_example_is_a_real_tool_call_type():
    examples = await generate_positive_examples()
    for ex in examples:
        assert ex.completion.type == "tool_call"
        assert len(ex.completion.tool_calls) == 1
        assert ex.source == "positive_template"


@pytest.mark.asyncio
async def test_all_offered_tools_are_real_and_covered_by_the_agent_mode_schema():
    examples = await generate_positive_examples(mode="agent")
    for ex in examples:
        offered_names = {t["function"]["name"] for t in ex.tools}
        assert ex.tool_family_holdout in offered_names

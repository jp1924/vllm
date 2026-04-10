import os

import pytest
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.reasoning import ReasoningParser, ReasoningParserManager

from tests.reasoning.utils import (
    StreamingReasoningReconstructor,
    run_reasoning_extraction,
    run_reasoning_extraction_streaming,
)
from transformers import AutoTokenizer


PARSER_NAME = "hyperclovax"

TOKENIZER_NAME = "naver-hyperclovax/HyperCLOVAX-SEED-Think-32B"


THINK_START = "/think\n"
THINK_END_BASE = "<|im_end|>\n<|im_start|>assistant"
FUNCTION_CALL_ROLE = " -> tool/function_call\n"


def _tool_payload(name: str = "search", args: str = '{"query":"weather"}') -> str:
    return f'[{{"name":"{name}","arguments":{args}}}]'


@pytest.fixture(scope="module")
def hyperclovax_tokenizer():
    local_candidates = [os.environ.get("HCX_TOKENIZER_PATH"), TOKENIZER_NAME]
    for path in local_candidates:
        if path and os.path.isdir(path):
            return AutoTokenizer.from_pretrained(path, local_files_only=True)

    pytest.skip(
        "Local HyperCLOVAX tokenizer is required. Set HCX_TOKENIZER_PATH or "
        "place tokenizer under naver-hyperclovax/HyperCLOVAX-SEED-Think-32B"
    )


@pytest.fixture
def parser(hyperclovax_tokenizer) -> ReasoningParser:
    return ReasoningParserManager.get_reasoning_parser(PARSER_NAME)(hyperclovax_tokenizer)


def test_hyperclovax_reasoning_parser_creation(hyperclovax_tokenizer):
    parser_cls = ReasoningParserManager.get_reasoning_parser(PARSER_NAME)
    created = parser_cls(hyperclovax_tokenizer)
    assert isinstance(created, ReasoningParser)


@pytest.fixture
def request_auto() -> ChatCompletionRequest:
    return ChatCompletionRequest(messages=[], model="test-model", tool_choice=None)


REASONING_WITH_CONTENT = {
    "output": THINK_START + "This is reasoning.\n" + THINK_END_BASE + "\nThis is the answer.",
    "reasoning": "This is reasoning.\n",
    "content": "\nThis is the answer.",
}

REASONING_ONLY = {
    "output": THINK_START + "Still thinking...",
    "reasoning": "Still thinking...",
    "content": None,
}

EMPTY_THINKING_NONSTREAM = {
    "output": THINK_START + THINK_END_BASE + "\nAnswer.",
    "reasoning": None,
    "content": "\nAnswer.",
}

NO_THINKING_NONSTREAM = {
    "output": "\nDirect answer.",
    "reasoning": None,
    "content": "Direct answer.",
    "tool_choice": "auto",
}

TOOL_CALL_AFTER_THINK_NONSTREAM = {
    "output": THINK_START + "Let me check.\n" + THINK_END_BASE + FUNCTION_CALL_ROLE + _tool_payload(),
    "reasoning": "Let me check.\n",
    "content": FUNCTION_CALL_ROLE + _tool_payload(),
}

DIRECT_TOOL_CALL_NONSTREAM = {
    "output": FUNCTION_CALL_ROLE + _tool_payload(),
    "reasoning": None,
    "content": _tool_payload(),
    "tool_choice": "required",
}

MULTILINE_REASONING = {
    "output": THINK_START + "Line one.\nLine two.\n" + THINK_END_BASE + "\nFinal answer.",
    "reasoning": "Line one.\nLine two.\n",
    "content": "\nFinal answer.",
}

NON_STREAMING_TEST_CASES = [
    pytest.param(REASONING_WITH_CONTENT, id="reasoning_with_content"),
    pytest.param(REASONING_ONLY, id="reasoning_only"),
    pytest.param(EMPTY_THINKING_NONSTREAM, id="empty_thinking"),
    pytest.param(NO_THINKING_NONSTREAM, id="no_thinking"),
    pytest.param(TOOL_CALL_AFTER_THINK_NONSTREAM, id="tool_call_after_think"),
    pytest.param(DIRECT_TOOL_CALL_NONSTREAM, id="direct_tool_call"),
    pytest.param(MULTILINE_REASONING, id="multiline_reasoning"),
]


EMPTY_THINKING_STREAM = {
    "output": THINK_START + THINK_END_BASE + "\nAnswer.",
    "reasoning": "",
    "content": "\nAnswer.",
}

NO_THINKING_STREAM = {
    "output": "\nDirect answer.",
    "reasoning": None,
    "content": "\nDirect answer.",
}

TOOL_CALL_AFTER_THINK_STREAM = {
    "output": THINK_START + "Let me check.\n" + THINK_END_BASE + FUNCTION_CALL_ROLE + _tool_payload(),
    "reasoning": "Let me check.\n",
    "content": _tool_payload(),
}

STREAMING_TEST_CASES = [
    pytest.param(REASONING_WITH_CONTENT, id="reasoning_with_content"),
    pytest.param(REASONING_ONLY, id="reasoning_only"),
    pytest.param(EMPTY_THINKING_STREAM, id="empty_thinking"),
    pytest.param(NO_THINKING_STREAM, id="no_thinking"),
    pytest.param(TOOL_CALL_AFTER_THINK_STREAM, id="tool_call_after_think"),
    pytest.param(MULTILINE_REASONING, id="multiline_reasoning"),
]


def _make_request(tool_choice=None) -> ChatCompletionRequest:
    if tool_choice in (None, "none"):
        return ChatCompletionRequest(messages=[], model="test-model")
    return ChatCompletionRequest(
        messages=[],
        model="test-model",
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "test tool",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        tool_choice=tool_choice,
    )


@pytest.mark.parametrize("param_dict", NON_STREAMING_TEST_CASES)
def test_extract_reasoning_nonstreaming(
    param_dict: dict,
    parser: ReasoningParser,
):
    tool_choice = param_dict.get("tool_choice", "none")
    request = _make_request(tool_choice=tool_choice)

    output_tokens = [
        parser.model_tokenizer.convert_tokens_to_string([tok])
        for tok in parser.model_tokenizer.tokenize(param_dict["output"])
    ]
    reasoning, content = run_reasoning_extraction(parser, output_tokens, request=request, streaming=False)

    assert reasoning == param_dict["reasoning"]
    assert content == param_dict["content"]


@pytest.mark.parametrize("param_dict", STREAMING_TEST_CASES)
def test_extract_reasoning_streaming(
    param_dict: dict,
    hyperclovax_tokenizer,
):
    fresh_parser = ReasoningParserManager.get_reasoning_parser(PARSER_NAME)(hyperclovax_tokenizer)

    output_tokens = [
        hyperclovax_tokenizer.convert_tokens_to_string([tok])
        for tok in hyperclovax_tokenizer.tokenize(param_dict["output"])
    ]
    reasoning, content = run_reasoning_extraction(fresh_parser, output_tokens, streaming=True)

    assert reasoning == param_dict["reasoning"]
    assert content == param_dict["content"]


def test_is_reasoning_end_true_with_newline_variant(parser: ReasoningParser):
    ids = parser.model_tokenizer.encode(THINK_START + "hello" + THINK_END_BASE + "\n")
    assert parser.is_reasoning_end(ids) is True


def test_is_reasoning_end_true_with_content_after_end(parser: ReasoningParser):
    ids = parser.model_tokenizer.encode(THINK_START + "hello" + THINK_END_BASE + FUNCTION_CALL_ROLE)
    assert parser.is_reasoning_end(ids) is True


def test_is_reasoning_end_false_start_after_end(parser: ReasoningParser):
    ids = parser.model_tokenizer.encode(THINK_END_BASE + "\n" + THINK_START + "more")
    assert parser.is_reasoning_end(ids) is False


def test_is_reasoning_end_false_no_end_token(parser: ReasoningParser):
    regular_ids = parser.model_tokenizer.encode("hello world, still reasoning")
    assert parser.is_reasoning_end(regular_ids) is False


def test_is_reasoning_end_true_single_end_token(parser: ReasoningParser):
    assert parser.is_reasoning_end([parser.end_token_id]) is True


def test_is_reasoning_end_streaming_true_on_end_token_delta(parser: ReasoningParser):
    assert parser.is_reasoning_end_streaming([parser.end_token_id], [parser.end_token_id]) is True


def test_is_reasoning_end_streaming_false_without_end_token_delta(
    parser: ReasoningParser,
):
    assert parser.is_reasoning_end_streaming([parser.end_token_id], [parser.end_token_id + 1]) is False


def test_is_reasoning_end_false_empty_sequence(parser: ReasoningParser):
    assert parser.is_reasoning_end([]) is False


def test_extract_content_ids_after_end_token(parser: ReasoningParser):
    sep_text = THINK_START + "abc" + THINK_END_BASE + "hello"
    all_ids = parser.model_tokenizer.encode(sep_text)
    content_ids = parser.extract_content_ids(all_ids)

    decoded = parser.model_tokenizer.decode(content_ids, skip_special_tokens=False)
    assert "hello" in decoded


def test_extract_content_ids_no_end_token(parser: ReasoningParser):
    still_reasoning_ids = parser.model_tokenizer.encode("still thinking")
    assert parser.extract_content_ids(still_reasoning_ids) == []


MULTI_TOKEN_DELTA_CASES = [
    pytest.param(
        [THINK_START + "reasoning", THINK_END_BASE + "content"],
        "reasoning",
        "content",
        id="end_tag_and_content_in_one_delta",
    ),
    pytest.param(
        [THINK_START + "start of thinking", " more", THINK_END_BASE + "ok"],
        "start of thinking more",
        "ok",
        id="start_marker_with_reasoning",
    ),
    pytest.param(
        [THINK_START + "reasoning", "<|im_end|>", "\n<|im_start|>assistant", "result"],
        "reasoning",
        "result",
        id="end_tag_split_across_deltas",
    ),
    pytest.param(
        ["\ndirect content"],
        None,
        "\ndirect content",
        id="no_thinking_single_delta",
    ),
    pytest.param(
        [THINK_START + "think", THINK_END_BASE + FUNCTION_CALL_ROLE + _tool_payload()],
        "think",
        _tool_payload(),
        id="tool_call_after_reasoning",
    ),
]


@pytest.mark.parametrize(
    "deltas, expected_reasoning, expected_content",
    MULTI_TOKEN_DELTA_CASES,
)
def test_streaming_multi_token_deltas(
    deltas: list[str],
    expected_reasoning: str | None,
    expected_content: str | None,
    hyperclovax_tokenizer,
):
    fresh_parser = ReasoningParserManager.get_reasoning_parser(PARSER_NAME)(hyperclovax_tokenizer)
    reconstructor: StreamingReasoningReconstructor = run_reasoning_extraction_streaming(fresh_parser, deltas)

    assert reconstructor.reasoning == expected_reasoning
    assert (reconstructor.other_content or None) == expected_content


def test_force_reasoning_treats_all_as_reasoning(parser: ReasoningParser):
    request = ChatCompletionRequest(
        messages=[],
        model="test-model",
        chat_template_kwargs={"force_reasoning": True},
    )
    reasoning, content = parser.extract_reasoning("No think marker but forced.", request)
    assert reasoning == "No think marker but forced."
    assert content is None


def test_skip_reasoning_returns_all_as_content(parser: ReasoningParser):
    request = ChatCompletionRequest(
        messages=[],
        model="test-model",
        chat_template_kwargs={"skip_reasoning": True},
    )
    reasoning, content = parser.extract_reasoning(THINK_START + "This should be content.", request)
    assert reasoning is None
    assert content == THINK_START + "This should be content."


def test_force_reasoning_takes_priority_over_skip(parser: ReasoningParser):
    request = ChatCompletionRequest(
        messages=[],
        model="test-model",
        chat_template_kwargs={"force_reasoning": True, "skip_reasoning": True},
    )
    reasoning, content = parser.extract_reasoning("some output", request)
    assert reasoning == "some output"
    assert content is None

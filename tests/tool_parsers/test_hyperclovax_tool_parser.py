# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import os

import pytest
from transformers import AutoTokenizer

from tests.tool_parsers.common_tests import ToolParserTestConfig, ToolParserTests
from tests.tool_parsers.utils import run_tool_extraction
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.hyperclovax_tool_parser import HyperCLOVAXToolParser

TOKENIZER_NAME = "naver-hyperclovax/HyperCLOVAX-SEED-Think-32B"

TOOL_CALL_START = " -> tool/function_call\n"
TOOL_CALL_END = "<|im_end|>"
ASSISTANT_TOOL_PREFIX = "<|im_end|>\n<|im_start|>assistant -> tool/function_call\n"


def _call(name: str, arguments: dict | None = None) -> dict:
    return {"name": name, "arguments": arguments or {}}


def _tool_call_output(
    calls: list[dict],
    *,
    with_end: bool = True,
    with_assistant_prefix: bool = False,
    leading_content: str = "",
) -> str:
    prefix = ASSISTANT_TOOL_PREFIX if with_assistant_prefix else TOOL_CALL_START
    payload = json.dumps(calls, ensure_ascii=False)
    suffix = TOOL_CALL_END if with_end else ""
    return leading_content + prefix + payload + suffix


@pytest.fixture(scope="module")
def hcx_tokenizer() -> TokenizerLike:
    local_candidates = [
        os.environ.get("HCX_TOKENIZER_PATH"),
        "/home/jp/DEMO/LLM42/base_models/HCX/HCX-SEED-Think-32B",
    ]
    for path in local_candidates:
        if path and os.path.isdir(path):
            return AutoTokenizer.from_pretrained(path, local_files_only=True)

    pytest.skip(
        "Local HyperCLOVAX tokenizer is required. Set HCX_TOKENIZER_PATH or "
        "place tokenizer under /home/jp/DEMO/LLM42/base_models/HCX/HCX-SEED-Think-32B"
    )


class TestHyperCLOVAXToolParser(ToolParserTests):
    @pytest.fixture
    def tokenizer(self, hcx_tokenizer: TokenizerLike) -> TokenizerLike:
        return hcx_tokenizer

    @pytest.fixture
    def test_config(self) -> ToolParserTestConfig:
        return ToolParserTestConfig(
            parser_name="hyperclovax",
            no_tool_calls_output="This is a plain response without any tools.",
            single_tool_call_output=_tool_call_output(
                [_call("get_weather", {"city": "Tokyo"})]
            ),
            parallel_tool_calls_output=_tool_call_output(
                [
                    _call("get_weather", {"city": "Tokyo"}),
                    _call("get_time", {"timezone": "Asia/Tokyo"}),
                ]
            ),
            various_data_types_output=(
                _tool_call_output(
                    [
                        _call(
                            "test_function",
                            {
                                "string_field": "hello",
                                "int_field": 42,
                                "float_field": 3.14,
                                "bool_field": True,
                                "null_field": None,
                                "array_field": ["a", "b", "c"],
                                "object_field": {"nested": "value"},
                                "empty_array": [],
                                "empty_object": {},
                            },
                        )
                    ]
                )
            ),
            empty_arguments_output=_tool_call_output([_call("refresh", {})]),
            surrounding_text_output=(
                _tool_call_output(
                    [_call("get_weather", {"city": "Tokyo"})],
                    with_assistant_prefix=True,
                    leading_content="I will call a tool.\n",
                )
            ),
            escaped_strings_output=(
                _tool_call_output(
                    [
                        _call(
                            "test_function",
                            {
                                "quoted": 'He said "hello"',
                                "path": "C:\\Users\\file.txt",
                                "newline": "line1\nline2",
                            },
                        )
                    ]
                )
            ),
            malformed_input_outputs=[
                TOOL_CALL_START + "[",
                TOOL_CALL_START + "not-json" + TOOL_CALL_END,
            ],
            single_tool_call_expected_name="get_weather",
            single_tool_call_expected_args={"city": "Tokyo"},
            single_tool_call_expected_content=None,
            parallel_tool_calls_count=2,
            parallel_tool_calls_names=["get_weather", "get_time"],
        )

    @pytest.mark.parametrize("streaming", [True, False])
    def test_tool_call_after_assistant_separator(
        self, hcx_tokenizer: TokenizerLike, streaming: bool
    ):
        model_output = _tool_call_output(
            [_call("get_weather", {"city": "Seoul"})],
            with_assistant_prefix=True,
            leading_content="Let me check.\n",
        )
        parser = HyperCLOVAXToolParser(hcx_tokenizer)
        content, tool_calls = run_tool_extraction(
            parser, model_output, streaming=streaming
        )

        assert len(tool_calls) == 1
        assert tool_calls[0].function.name == "get_weather"
        assert json.loads(tool_calls[0].function.arguments) == {"city": "Seoul"}
        if not streaming:
            assert content == "Let me check.\n"

    @pytest.mark.parametrize("streaming", [True, False])
    def test_no_tool_call_plain_text(
        self, hcx_tokenizer: TokenizerLike, streaming: bool
    ):
        model_output = "\nThis is a regular assistant reply."
        parser = HyperCLOVAXToolParser(hcx_tokenizer)
        content, tool_calls = run_tool_extraction(
            parser, model_output, streaming=streaming
        )

        assert len(tool_calls) == 0
        assert content == model_output

    @pytest.mark.parametrize("streaming", [True, False])
    def test_multiple_tool_calls_extracted_in_order(
        self, hcx_tokenizer: TokenizerLike, streaming: bool
    ):
        model_output = _tool_call_output(
            [
                _call("alpha", {"x": 1}),
                _call("beta", {"y": 2}),
                _call("gamma", {"z": 3}),
            ]
        )
        parser = HyperCLOVAXToolParser(hcx_tokenizer)
        content, tool_calls = run_tool_extraction(
            parser, model_output, streaming=streaming
        )

        assert [tc.function.name for tc in tool_calls] == ["alpha", "beta", "gamma"]
        assert json.loads(tool_calls[0].function.arguments) == {"x": 1}
        assert json.loads(tool_calls[1].function.arguments) == {"y": 2}
        assert json.loads(tool_calls[2].function.arguments) == {"z": 3}

    @pytest.mark.parametrize("streaming", [True, False])
    def test_nested_arguments_preserved(
        self, hcx_tokenizer: TokenizerLike, streaming: bool
    ):
        model_output = _tool_call_output(
            [
                _call(
                    "create_event",
                    {
                        "title": "Meeting",
                        "location": {"city": "Seoul", "room": "A1"},
                        "attendees": ["alice", "bob"],
                    },
                )
            ]
        )
        parser = HyperCLOVAXToolParser(hcx_tokenizer)
        content, tool_calls = run_tool_extraction(
            parser, model_output, streaming=streaming
        )

        assert len(tool_calls) == 1
        args = json.loads(tool_calls[0].function.arguments)
        assert args["location"] == {"city": "Seoul", "room": "A1"}
        assert args["attendees"] == ["alice", "bob"]

    @pytest.mark.parametrize("streaming", [True, False])
    def test_unicode_in_arguments_preserved(
        self, hcx_tokenizer: TokenizerLike, streaming: bool
    ):
        model_output = _tool_call_output([_call("greet", {"message": "hello"})])
        parser = HyperCLOVAXToolParser(hcx_tokenizer)
        content, tool_calls = run_tool_extraction(
            parser, model_output, streaming=streaming
        )

        assert len(tool_calls) == 1
        args = json.loads(tool_calls[0].function.arguments)
        assert args["message"] == "hello"

    def test_each_streaming_tool_call_has_unique_id(self, hcx_tokenizer: TokenizerLike):
        model_output = _tool_call_output([_call("func_a", {}), _call("func_b", {})])
        parser = HyperCLOVAXToolParser(hcx_tokenizer)
        _, tool_calls = run_tool_extraction(parser, model_output, streaming=True)

        assert len(tool_calls) == 2
        ids = [tc.id for tc in tool_calls]
        assert all(ids), "All tool call IDs must be non-empty"
        assert len(set(ids)) == len(ids), "Tool call IDs must be unique"

    def test_malformed_json_returns_no_tool_calls(self, hcx_tokenizer: TokenizerLike):
        model_output = TOOL_CALL_START + "[{not valid json}]" + TOOL_CALL_END
        parser = HyperCLOVAXToolParser(hcx_tokenizer)
        result = parser.extract_tool_calls(model_output, request=None)
        assert not result.tools_called
        assert result.tool_calls == []
        assert result.content == model_output

    def test_missing_end_token_still_parsed(self, hcx_tokenizer: TokenizerLike):
        model_output = _tool_call_output(
            [_call("search", {"top_k": 3})], with_end=False
        )
        parser = HyperCLOVAXToolParser(hcx_tokenizer)
        result = parser.extract_tool_calls(model_output, request=None)

        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "search"
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args == {"top_k": 3}

    def test_no_marker_returns_content_unchanged(self, hcx_tokenizer: TokenizerLike):
        model_output = "\nHello, how can I help you?"
        parser = HyperCLOVAXToolParser(hcx_tokenizer)
        result = parser.extract_tool_calls(model_output, request=None)
        assert not result.tools_called
        assert result.content == model_output

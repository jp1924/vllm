# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
from collections.abc import Sequence

import regex as re

from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import Tool, ToolParser

logger = init_logger(__name__)


class HyperCLOVAXToolParser(ToolParser):
    def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
        super().__init__(tokenizer, tools)

        self.tool_call_start_token: str = " -> tool/function_call\n"
        self.tool_call_end_token: str = "<|im_end|>"
        self.tool_call_regex = re.compile(
            r"-> tool/function_call\n(.*?)<\|im_end\|>|"
            r"-> tool/function_call\n(.*)]",
            re.DOTALL,
        )

        self.tool_call_offset = 0
        self._buffer = ""
        self._sent_content_len = 0
        self._pending_messages: list[DeltaMessage] = []

    @staticmethod
    def _partial_tag_overlap(text: str, tag: str) -> int:
        max_check = min(len(tag) - 1, len(text))
        for k in range(max_check, 0, -1):
            if text.endswith(tag[:k]):
                return k
        return 0

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        if self.tool_call_start_token not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output,
            )

        try:
            tool_call_match = self.tool_call_regex.search(model_output)
            if tool_call_match is None:
                return ExtractedToolCallInformation(
                    tools_called=False,
                    tool_calls=[],
                    content=model_output,
                )

            if tool_call_match.group(1) is not None:
                raw_function_calls = json.loads(tool_call_match.group(1))
            else:
                raw_function_calls = json.loads(tool_call_match.group(2) + "]")

            if isinstance(raw_function_calls, dict):
                raw_function_calls = [raw_function_calls]
            if not isinstance(raw_function_calls, list):
                raise ValueError("tool calls payload must be object or list")

            tool_calls = [
                ToolCall(
                    type="function",
                    function=FunctionCall(
                        name=function_call["name"],
                        arguments=json.dumps(
                            function_call["arguments"], ensure_ascii=False
                        ),
                    ),
                )
                for function_call in raw_function_calls
            ]

            prefix = "<|im_end|>\n<|im_start|>assistant -> tool/function_call\n"
            if prefix in model_output:
                content = model_output.split(prefix)[0]
                return ExtractedToolCallInformation(
                    tools_called=True,
                    tool_calls=tool_calls,
                    content=content if content else None,
                )

            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=None,
            )

        except Exception:
            logger.exception("Error extracting tool call from response.")
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output,
            )

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        if self._pending_messages:
            return self._pending_messages.pop(0)

        self._buffer += delta_text

        if self.tool_call_start_token not in self._buffer:
            overlap = self._partial_tag_overlap(
                self._buffer, self.tool_call_start_token
            )
            safe_len = len(self._buffer) - overlap
            if safe_len > self._sent_content_len:
                content = self._buffer[self._sent_content_len : safe_len]
                self._sent_content_len = safe_len
                return DeltaMessage(content=content)
            return None

        marker_idx = self._buffer.find(self.tool_call_start_token)
        if self._sent_content_len < marker_idx:
            content = self._buffer[self._sent_content_len : marker_idx]
            self._sent_content_len = marker_idx
            if content:
                return DeltaMessage(content=content)

        if marker_idx + len(self.tool_call_start_token) > len(self._buffer):
            return None

        function_call_text = self._buffer[
            marker_idx + len(self.tool_call_start_token) :
        ]
        function_call_text = function_call_text[self.tool_call_offset :]

        opening_brace_index = None
        for idx, ch in enumerate(function_call_text):
            if ch == "{":
                opening_brace_index = idx
                break

        if opening_brace_index is None:
            return None

        closing_brace_indices = [
            idx for idx, ch in enumerate(function_call_text) if ch == "}"
        ]
        if not closing_brace_indices:
            return None

        for closing_brace_index in closing_brace_indices:
            candidate = function_call_text[
                opening_brace_index : closing_brace_index + 1
            ]
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue

            if not isinstance(parsed, dict):
                continue

            self.current_tool_id += 1
            self.tool_call_offset += closing_brace_index + 1
            self.prev_tool_call_arr.append(parsed)
            self.streamed_args_for_tool.append(candidate)

            return DeltaMessage(
                tool_calls=[
                    DeltaToolCall(
                        index=self.current_tool_id,
                        type="function",
                        id=make_tool_call_id(),
                        function=DeltaFunctionCall(
                            name=parsed.get("name", ""),
                            arguments=json.dumps(
                                parsed.get("arguments", ""), ensure_ascii=False
                            ),
                        ).model_dump(exclude_none=True),
                    )
                ]
            )

        return None

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

import regex as re

from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.logger import init_logger
from vllm.reasoning.abs_reasoning_parsers import ReasoningParser
from vllm.tokenizers import TokenizerLike

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest

logger = init_logger(__name__)


class HyperCLOVAXReasoningParser(ReasoningParser):
    def __init__(self, tokenizer: TokenizerLike):
        super().__init__(tokenizer)

        self.think_start_token = "/think\n"
        self.think_end_string_base = "<|im_end|>\n<|im_start|>assistant"
        self.function_call_role = " -> tool/function_call\n"

        self.end_token_id = self.vocab.get("<|im_end|>")
        self.non_reasoning_mode_start_token = tokenizer.encode("\n")[0]
        self.no_reasoning_content = False

        self.exact_think_end_strings = [
            self.think_end_string_base + "\n",
            self.think_end_string_base + self.function_call_role,
        ]
        self.think_end_tokens = [
            tokenizer.encode(think_end_string)
            for think_end_string in self.exact_think_end_strings
        ]

        self.buffer_string = ""
        self.special_strings = [
            self.think_start_token,
            self.think_end_string_base,
            self.function_call_role,
        ]
        self.escaped_special_strings = [re.escape(ss) for ss in self.special_strings]

    def _remove_special_string(self) -> tuple[str, str]:
        positions: list[tuple[int, int]] = []
        for ss in self.escaped_special_strings:
            positions += [
                (m.start(), m.end()) for m in re.finditer(ss, self.buffer_string)
            ]

        sorted_positions = sorted(positions, key=lambda x: x[0])
        to_stream = self.buffer_string[: sorted_positions[-1][0]]
        remaining = self.buffer_string[sorted_positions[-1][1] :]
        for ss in self.special_strings:
            to_stream = to_stream.replace(ss, "")

        return to_stream, remaining

    def _check_is_special_string(self) -> bool:
        return any(ss in self.buffer_string for ss in self.special_strings)

    def _check_is_part_of_special_string(self) -> bool:
        for ss in self.special_strings:
            min_len = min(len(self.buffer_string), len(ss))
            for ln in range(min_len, 0, -1):
                if self.buffer_string[-ln:] == ss[:ln]:
                    return True
        return False

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        if len(input_ids) > 1:
            for think_end_tokens in self.think_end_tokens:
                think_end_len = len(think_end_tokens)
                if (
                    len(input_ids) >= think_end_len
                    and input_ids[-think_end_len:] == think_end_tokens
                ):
                    return True
            return False

        return self.no_reasoning_content or (
            self.end_token_id is not None and self.end_token_id in input_ids
        )

    def is_reasoning_end_streaming(
        self, input_ids: Sequence[int], delta_ids: Iterable[int]
    ) -> bool:
        if self.end_token_id is None:
            return False
        return self.end_token_id in delta_ids

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        if self.end_token_id is None or self.end_token_id not in input_ids[:-1]:
            return []
        return input_ids[input_ids.index(self.end_token_id) + 1 :]

    def extract_reasoning(
        self, model_output: str, request: "ChatCompletionRequest | ResponsesRequest"
    ) -> tuple[str | None, str | None]:
        chat_template_kwargs = getattr(request, "chat_template_kwargs", None) or {}
        tool_choice = getattr(request, "tool_choice", None)

        is_reasoning = False
        if chat_template_kwargs.get("force_reasoning", False):
            is_reasoning = True
        elif chat_template_kwargs.get("skip_reasoning", False):
            return None, model_output

        if model_output.startswith(self.think_start_token):
            is_reasoning = True
            _, _, model_output = model_output.partition(self.think_start_token)

        if self.think_end_string_base not in model_output:
            if is_reasoning:
                return model_output, None

            if tool_choice in ("auto", None):
                if model_output.startswith("\n"):
                    model_output = model_output[1:]
                return None, model_output

            return None, model_output.replace(self.function_call_role, "")

        reasoning_content, _, content = model_output.partition(
            self.think_end_string_base
        )
        return reasoning_content or None, content or None

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        if current_token_ids and (
            current_token_ids[0] == self.non_reasoning_mode_start_token
        ):
            self.no_reasoning_content = True

        if len(current_text) == 0:
            return None

        if self.no_reasoning_content:
            return DeltaMessage(content=delta_text)

        self.buffer_string += delta_text

        if self._check_is_special_string():
            if current_text.startswith(self.function_call_role):
                self.no_reasoning_content = True
                delta_text = self.buffer_string
                self.buffer_string = ""
                return DeltaMessage(content=delta_text)

            buffered_content, delta_text = self._remove_special_string()
            self.buffer_string = delta_text

            if buffered_content:
                if self._check_is_part_of_special_string():
                    return DeltaMessage(reasoning=buffered_content)
                self.buffer_string = ""
                return DeltaMessage(reasoning=buffered_content, content=delta_text)

        if self._check_is_part_of_special_string():
            return None

        delta_text = self.buffer_string
        self.buffer_string = ""

        if self.think_end_string_base in current_text:
            return DeltaMessage(content=delta_text)
        return DeltaMessage(reasoning=delta_text)

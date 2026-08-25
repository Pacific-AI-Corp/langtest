from __future__ import annotations

from enum import Enum
from typing import Any, Union

from pydantic import Field

from .content import SchemaBase, TextContent
from .message import Message
from .tool import (
    FunctionCall,
    FunctionCallOutput,
    ToolCall,
    ToolResult,
)


class FinishReason(str, Enum):
    STOP = "stop"
    LENGTH = "length"
    TOOL_CALL = "tool_call"
    CONTENT_FILTER = "content_filter"
    MAX_TOKENS = "max_tokens"
    ERROR = "error"
    OTHER = "other"


class ReasoningItem(SchemaBase):
    """
    Reasoning metadata/item.

    The actual reasoning content should remain opaque unless the provider
    explicitly exposes it.
    """

    type: str = "reasoning"

    id: str | None = None

    summary: list[Any] = Field(default_factory=list)

    encrypted_content: str | None = None


class UnknownOutputItem(SchemaBase):
    """
    Escape hatch for provider-specific output items.
    """

    type: str

    raw: dict[str, Any] = Field(default_factory=dict)


OutputItem = Union[
    Message,
    FunctionCall,
    FunctionCallOutput,
    ToolCall,
    ToolResult,
    ReasoningItem,
    UnknownOutputItem,
]


class Output(SchemaBase):
    """
    Canonical model output.

    `items` is the authoritative representation.

    `text` is a convenience projection and should not be treated
    as the complete response.
    """

    items: list[OutputItem] = Field(default_factory=list)

    text: str | None = None

    id: str | None = None

    response_id: str | None = None

    model: str | None = None

    provider: str | None = None

    finish_reason: FinishReason | str | None = None

    usage: dict[str, Any] | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)

    # Original provider response.
    raw: Any = None

    @property
    def output_text(self) -> str:
        """
        Return the textual portion of the output.

        Tool calls and reasoning items are not converted to text.
        """

        if self.text is not None:
            return self.text

        chunks: list[str] = []

        for item in self.items:

            if not isinstance(item, Message):
                continue

            if isinstance(item.content, str):
                chunks.append(item.content)
                continue

            for content in item.content:

                if isinstance(
                    content,
                    TextContent,
                ):
                    chunks.append(content.text)

        return "".join(chunks)

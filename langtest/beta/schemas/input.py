from __future__ import annotations

from typing import Any, Union

from pydantic import Field, model_validator

from .content import SchemaBase
from .message import Message
from .tool import (
    FunctionCallOutput,
    ToolResult,
)


class UnknownInputItem(SchemaBase):
    """
    Escape hatch for provider-specific input items.

    The beta module should not fail just because a provider introduces
    a new input item type that LangTest does not explicitly model yet.
    """

    type: str

    raw: dict[str, Any] = Field(default_factory=dict)


InputItem = Union[
    Message,
    FunctionCallOutput,
    ToolResult,
    UnknownInputItem,
]


class Input(SchemaBase):
    """
    Canonical input to a model/client.

    Simple request:

        Input(text="Hello")

    Conversational request:

        Input(
            items=[
                Message(
                    role="user",
                    content="Hello",
                )
            ]
        )

    Agent/tool request:

        Input(
            items=[
                FunctionCallOutput(
                    call_id="call_123",
                    output="72°F",
                )
            ]
        )
    """

    text: str | None = None

    items: list[InputItem] = Field(default_factory=list)

    # Useful for stateful APIs such as OpenAI Responses.
    previous_response_id: str | None = None

    conversation_id: str | None = None

    # Optional metadata for LangTest.
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_input(self) -> "Input":

        if self.text is None and not self.items:
            raise ValueError("Input requires either `text` or `items`.")

        if self.text is not None and self.items:
            raise ValueError("Input cannot contain both `text` and `items`.")

        return self

    @classmethod
    def from_text(
        cls,
        text: str,
    ) -> "Input":

        return cls(text=text)

    @classmethod
    def from_message(
        cls,
        message: Message,
    ) -> "Input":

        return cls(items=[message])

    @classmethod
    def from_items(
        cls,
        items: list[InputItem],
    ) -> "Input":

        return cls(items=items)

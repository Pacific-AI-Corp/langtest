from __future__ import annotations

from enum import Enum

from .content import Content
from .content import SchemaBase


class Role(str, Enum):
    SYSTEM = "system"
    DEVELOPER = "developer"
    USER = "user"
    ASSISTANT = "assistant"
    MODEL = "model"
    TOOL = "tool"


class Message(SchemaBase):
    """
    Canonical conversational message.

    Supports both simple text:

        Message(
            role="user",
            content="Hello",
        )

    and multimodal content:

        Message(
            role="user",
            content=[
                TextContent(...),
                ImageContent(...),
            ],
        )
    """

    type: str = "message"

    role: Role | str

    content: str | list[Content]

    name: str | None = None

    id: str | None = None

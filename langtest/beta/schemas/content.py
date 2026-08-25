from __future__ import annotations

from typing import Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field


class SchemaBase(BaseModel):
    """
    Base schema used by LangTest beta.

    `extra="allow"` is intentional. Providers introduce new fields and
    content types frequently, so the canonical schema should not break
    when an adapter encounters provider-specific metadata.
    """

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
    )


class TextContent(SchemaBase):
    type: Literal[
        "text",
        "input_text",
        "output_text",
    ] = "text"

    text: str


class ImageContent(SchemaBase):
    type: Literal[
        "image",
        "input_image",
        "image_url",
    ] = "image"

    image_url: str | None = None
    data: str | None = None
    media_type: str | None = None
    detail: str | None = None


class AudioContent(SchemaBase):
    type: Literal[
        "audio",
        "input_audio",
        "audio_url",
    ] = "audio"

    data: str | None = None
    media_type: str | None = None
    audio_url: str | None = None


class FileContent(SchemaBase):
    type: Literal[
        "file",
        "input_file",
        "file_url",
    ] = "file"

    file_id: str | None = None
    file_url: str | None = None
    filename: str | None = None
    data: str | None = None
    media_type: str | None = None


class RefusalContent(SchemaBase):
    type: Literal["refusal"] = "refusal"

    refusal: str


class ToolUseContent(SchemaBase):
    """
    Generic representation of a provider tool invocation.

    Useful for Anthropic-style content blocks and other providers
    that represent tool calls inside message content.
    """

    type: Literal["tool_use"] = "tool_use"

    id: str
    name: str

    input: dict[str, Any] = Field(default_factory=dict)


class ToolResultContent(SchemaBase):
    """
    Generic representation of a provider tool result.
    """

    type: Literal["tool_result"] = "tool_result"

    tool_use_id: str

    content: Any = None

    is_error: bool = False


Content = Union[
    TextContent,
    ImageContent,
    AudioContent,
    FileContent,
    RefusalContent,
    ToolUseContent,
    ToolResultContent,
]

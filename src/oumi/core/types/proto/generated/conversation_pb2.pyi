from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import (
    ClassVar as _ClassVar,
    Iterable as _Iterable,
    Mapping as _Mapping,
    Optional as _Optional,
    Union as _Union,
)

DESCRIPTOR: _descriptor.FileDescriptor

class Role(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ROLE_UNSPECIFIED: _ClassVar[Role]
    SYSTEM: _ClassVar[Role]
    USER: _ClassVar[Role]
    ASSISTANT: _ClassVar[Role]
    TOOL: _ClassVar[Role]

ROLE_UNSPECIFIED: Role
SYSTEM: Role
USER: Role
ASSISTANT: Role
TOOL: Role

class DataBlob(_message.Message):
    __slots__ = ("mime_type", "binary_data")
    MIME_TYPE_FIELD_NUMBER: _ClassVar[int]
    BINARY_DATA_FIELD_NUMBER: _ClassVar[int]
    mime_type: str
    binary_data: bytes
    def __init__(
        self, mime_type: _Optional[str] = ..., binary_data: _Optional[bytes] = ...
    ) -> None: ...

class ContentPart(_message.Message):
    __slots__ = ("type", "content", "blob")
    class Type(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        TYPE_UNSPECIFIED: _ClassVar[ContentPart.Type]
        TEXT: _ClassVar[ContentPart.Type]
        IMAGE_PATH: _ClassVar[ContentPart.Type]
        IMAGE_URL: _ClassVar[ContentPart.Type]
        IMAGE_BINARY: _ClassVar[ContentPart.Type]

    TYPE_UNSPECIFIED: ContentPart.Type
    TEXT: ContentPart.Type
    IMAGE_PATH: ContentPart.Type
    IMAGE_URL: ContentPart.Type
    IMAGE_BINARY: ContentPart.Type
    TYPE_FIELD_NUMBER: _ClassVar[int]
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    BLOB_FIELD_NUMBER: _ClassVar[int]
    type: ContentPart.Type
    content: str
    blob: DataBlob
    def __init__(
        self,
        type: _Optional[_Union[ContentPart.Type, str]] = ...,
        content: _Optional[str] = ...,
        blob: _Optional[_Union[DataBlob, _Mapping]] = ...,
    ) -> None: ...

class Message(_message.Message):
    __slots__ = ("id", "role", "parts")
    ID_FIELD_NUMBER: _ClassVar[int]
    ROLE_FIELD_NUMBER: _ClassVar[int]
    PARTS_FIELD_NUMBER: _ClassVar[int]
    id: str
    role: Role
    parts: _containers.RepeatedCompositeFieldContainer[ContentPart]
    def __init__(
        self,
        id: _Optional[str] = ...,
        role: _Optional[_Union[Role, str]] = ...,
        parts: _Optional[_Iterable[_Union[ContentPart, _Mapping]]] = ...,
    ) -> None: ...

class Conversation(_message.Message):
    __slots__ = ("conversation_id", "messages", "metadata")
    class MetadataEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(
            self, key: _Optional[str] = ..., value: _Optional[str] = ...
        ) -> None: ...

    CONVERSATION_ID_FIELD_NUMBER: _ClassVar[int]
    MESSAGES_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    conversation_id: str
    messages: _containers.RepeatedCompositeFieldContainer[Message]
    metadata: _containers.ScalarMap[str, str]
    def __init__(
        self,
        conversation_id: _Optional[str] = ...,
        messages: _Optional[_Iterable[_Union[Message, _Mapping]]] = ...,
        metadata: _Optional[_Mapping[str, str]] = ...,
    ) -> None: ...

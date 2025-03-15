from collections.abc import Iterable as _Iterable
from collections.abc import Mapping as _Mapping
from typing import ClassVar as _ClassVar
from typing import Optional as _Optional
from typing import Union as _Union

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper

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

class ContentItem(_message.Message):
    __slots__ = ("type", "content", "blob")
    class Type(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        TYPE_UNSPECIFIED: _ClassVar[ContentItem.Type]
        TEXT: _ClassVar[ContentItem.Type]
        IMAGE_PATH: _ClassVar[ContentItem.Type]
        IMAGE_URL: _ClassVar[ContentItem.Type]
        IMAGE_BINARY: _ClassVar[ContentItem.Type]

    TYPE_UNSPECIFIED: ContentItem.Type
    TEXT: ContentItem.Type
    IMAGE_PATH: ContentItem.Type
    IMAGE_URL: ContentItem.Type
    IMAGE_BINARY: ContentItem.Type
    TYPE_FIELD_NUMBER: _ClassVar[int]
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    BLOB_FIELD_NUMBER: _ClassVar[int]
    type: ContentItem.Type
    content: str
    blob: DataBlob
    def __init__(
        self,
        type: _Optional[_Union[ContentItem.Type, str]] = ...,
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
    parts: _containers.RepeatedCompositeFieldContainer[ContentItem]
    def __init__(
        self,
        id: _Optional[str] = ...,
        role: _Optional[_Union[Role, str]] = ...,
        parts: _Optional[_Iterable[_Union[ContentItem, _Mapping]]] = ...,
    ) -> None: ...

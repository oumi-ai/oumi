from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import (
    ClassVar as _ClassVar,
    Mapping as _Mapping,
    Optional as _Optional,
    Union as _Union,
)

DESCRIPTOR: _descriptor.FileDescriptor

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

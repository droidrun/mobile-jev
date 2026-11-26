"""Format tool failures before they enter memory and run records."""

from collections.abc import Callable, Mapping, Sequence
from types import UnionType
from typing import TYPE_CHECKING, Annotated, Literal, Union, cast, get_args, get_origin

from mobilerun.errors.base import is_value_safe
from pydantic import (
    AliasChoices,
    AliasPath,
    BaseModel,
    Discriminator,
    PydanticUserError,
    Tag,
    TypeAdapter,
    ValidationError,
)
from pydantic.fields import FieldInfo
from pydantic_core import ErrorDetails
from pydantic_core.core_schema import ErrorType

if TYPE_CHECKING:
    from typing_extensions import TypeForm

_BUILTIN_ERROR_TYPES = frozenset(get_args(ErrorType))
_SAFE_MESSAGES = {
    "assertion_error": "Assertion failed",
    "bytes_invalid_encoding": "Data has invalid encoding",
    "datetime_object_invalid": "Invalid datetime object",
    "get_attribute_error": "Error extracting attribute",
    "iteration_error": "Error iterating over input",
    "mapping_type": "Input should be a valid mapping",
    "no_such_attribute": "Unknown attribute",
    "timezone_offset": "Invalid timezone offset",
    "union_tag_invalid": "Input tag does not match any expected tag",
    "uuid_parsing": "Input should be a valid UUID",
    "value_error": "Value error",
}


def _branch_matches(
    annotation: object, tag: str | int, discriminator: str | Discriminator | None
) -> bool:
    if get_origin(annotation) is Annotated:
        inner, *metadata = get_args(annotation)
        for item in metadata:
            match item:
                case Tag(tag=label):
                    return str(tag) == label
    else:
        inner = annotation

    if discriminator is not None:
        match discriminator:
            case Discriminator(discriminator=str() as key) | (str() as key):
                match inner:
                    case type() as model if issubclass(model, BaseModel):
                        field = model.model_fields.get(key)
                        if field is not None and get_origin(field.annotation) is Literal:
                            return any(
                                str(tag) == str(value) for value in get_args(field.annotation)
                            )
        return False

    # Pydantic labels regular union locations with validator titles, including
    # composite labels such as list[Address] and function-after[...].
    try:
        return tag == TypeAdapter(cast("TypeForm[object]", annotation)).validator.title
    except PydanticUserError:
        return False


def _alias_paths(field: FieldInfo) -> list[list[str | int]]:
    paths: list[list[str | int]] = [[field.alias]] if field.alias is not None else []
    match field.validation_alias:
        case str() as alias:
            paths.append([alias])
        case AliasPath(path=path):
            paths.append(path)
        case AliasChoices() as choices:
            paths.extend(choices.convert_to_aliases())
    return paths


def _location(
    issue: ErrorDetails, schema: type[BaseModel] | None, protect: Callable[[str], None] | None
) -> str:
    parts: list[str] = []
    annotation: object = schema
    discriminator: str | Discriminator | None = None
    loc = issue["loc"]
    alias_end = 0
    for i, part in enumerate(loc):
        if i < alias_end:
            continue
        while get_origin(annotation) in (Annotated, Union, UnionType):
            if get_origin(annotation) is Annotated:
                annotation, *metadata = get_args(annotation)
                for item in metadata:
                    match item:
                        case FieldInfo(discriminator=value) if value is not None:
                            discriminator = value
                        case Discriminator():
                            discriminator = item
                continue
            branches = tuple(branch for branch in get_args(annotation) if branch is not type(None))
            if len(branches) == 1 and discriminator is None:
                annotation = branches[0]
                continue
            matches = [
                branch for branch in branches if _branch_matches(branch, part, discriminator)
            ]
            # An ambiguous or unrecognized tag must not turn any suffix into secrets.
            annotation = matches[0] if len(matches) == 1 else None
            discriminator = None
            break
        else:
            discriminator = None
            origin, args = get_origin(annotation), get_args(annotation)
            match annotation, part:
                case type() as model, str() as name if issubclass(model, BaseModel):
                    field = model.model_fields.get(name)
                    alias_end = i + 1
                    alias_root = False
                    if field is None:
                        for candidate in model.model_fields.values():
                            for path in _alias_paths(candidate):
                                if path[0] != name:
                                    continue
                                alias_root = True
                                end = i + len(path)
                                if loc[i:end] == tuple(path) and (field is None or end > alias_end):
                                    field = candidate
                                    alias_end = end
                    if field is not None:
                        parts.extend(str(segment) for segment in loc[i:alias_end])
                        annotation = field.annotation
                        discriminator = field.discriminator
                        continue
                    if (
                        not alias_root
                        and i == len(loc) - 1
                        and issue["type"] == "extra_forbidden"
                        and protect is not None
                    ):
                        protect(name)
                case _, int() as index if origin in (list, tuple, set, frozenset, Sequence):
                    parts.append(str(index))
                    if origin is tuple and args and args[-1] is not Ellipsis:
                        annotation = args[index] if 0 <= index < len(args) else None
                    else:
                        annotation = args[0] if args else None
                    continue
                case _ if origin in (dict, Mapping) or annotation in (dict, Mapping):
                    if protect is not None:
                        protect(str(part))
                    parts.append("*")
                    annotation = args[1] if len(args) == 2 else None
                    continue
            parts.append("*")
            annotation = None
    return ".".join(parts) or "output"


def tool_error_summary(
    error: Exception,
    schema: type[BaseModel] | None = None,
    protect: Callable[[str], None] | None = None,
) -> str:
    match error:
        case ValidationError():
            lines = []
            for issue in error.errors(
                include_input=False, include_url=False, include_context=False
            ):
                location = _location(issue, schema, protect)
                kind = issue["type"]
                message = (
                    _SAFE_MESSAGES.get(kind, issue["msg"])
                    if kind in _BUILTIN_ERROR_TYPES
                    else "Invalid value"
                )
                lines.append(f"{location}: {message}")
            return "\n".join(lines)
        case _:
            name = type(error).__name__
            return f"{name}: {error}" if is_value_safe(error) else name

# pyright: reportUnknownArgumentType=false
import logging
from typing import Any, Literal

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class StrictJsonSchemaObject(BaseModel):
    type: Literal["object"]
    additionalProperties: Literal[False]
    properties: dict[str, "StrictJsonSchema"]
    required: list[str]
    title: str | None = None

    @classmethod
    def init(cls, schema: dict[str, Any]) -> "StrictJsonSchemaObject":
        title = schema.get("title")
        properties_raw = schema.get("properties") or {}
        required = schema.get("required") or []

        strict_properties: dict[str, StrictJsonSchema] = {}
        for prop_name, prop_schema in properties_raw.items():
            strict_prop = init_strict_json_schema(prop_schema)
            if prop_name not in required:
                # Make optional fields nullable via anyOf
                strict_prop = StrictJsonSchemaAnyOf(
                    anyOf=[strict_prop, StrictJsonSchemaNull(type="null", title=None)],
                    title=None,
                )
            strict_properties[prop_name] = strict_prop

        return cls(
            type="object",
            additionalProperties=False,
            properties=strict_properties,
            required=list(strict_properties.keys()),
            title=title,
        )


class StrictJsonSchemaArray(BaseModel):
    type: Literal["array"]
    items: "StrictJsonSchema"
    minItems: int | None = None
    maxItems: int | None = None
    uniqueItems: bool | None = None
    title: str | None = None

    @classmethod
    def init(cls, schema: dict[str, Any]) -> "StrictJsonSchemaArray":
        title = schema.get("title")
        items_raw = schema.get("items")
        if items_raw is None:
            prefix_items = schema.get("prefixItems")
            min_items = schema.get("minItems")
            max_items = schema.get("maxItems")
            if (
                isinstance(prefix_items, list)
                and isinstance(min_items, int)
                and isinstance(max_items, int)
                and min_items == max_items == len(prefix_items)
                and len(prefix_items) > 0
            ):
                first = prefix_items[0]
                if all(item == first for item in prefix_items[1:]):
                    items_raw = first
        if items_raw is None:
            raise ValueError(
                "Array schema must define items, or provide uniform prefixItems with equal minItems/maxItems length."
            )

        strict_items = init_strict_json_schema(items_raw)

        return cls(
            type="array",
            items=strict_items,
            minItems=schema.get("minItems"),
            maxItems=schema.get("maxItems"),
            uniqueItems=schema.get("uniqueItems"),
            title=title,
        )


class StrictJsonSchemaString(BaseModel):
    type: Literal["string"]
    # Supported string restrictions
    minLength: int | None = None
    maxLength: int | None = None
    pattern: str | None = None
    enum: list[str] | None = None
    title: str | None = None

    @classmethod
    def init(cls, schema: dict[str, Any]) -> "StrictJsonSchemaString":
        return cls(
            type="string",
            minLength=schema.get("minLength"),
            maxLength=schema.get("maxLength"),
            pattern=schema.get("pattern"),
            enum=schema.get("enum"),
            title=schema.get("title"),
        )


class StrictJsonSchemaNumber(BaseModel):
    type: Literal["number"]
    minimum: float | None = None
    maximum: float | None = None
    exclusiveMinimum: float | None = None
    exclusiveMaximum: float | None = None
    multipleOf: float | None = None
    enum: list[float] | None = None
    title: str | None = None

    @classmethod
    def init(cls, schema: dict[str, Any]) -> "StrictJsonSchemaNumber":
        return cls(
            type="number",
            minimum=schema.get("minimum"),
            maximum=schema.get("maximum"),
            exclusiveMinimum=schema.get("exclusiveMinimum"),
            exclusiveMaximum=schema.get("exclusiveMaximum"),
            multipleOf=schema.get("multipleOf"),
            enum=schema.get("enum"),
            title=schema.get("title"),
        )


class StrictJsonSchemaInteger(BaseModel):
    type: Literal["integer"]
    minimum: int | None = None
    maximum: int | None = None
    exclusiveMinimum: int | None = None
    exclusiveMaximum: int | None = None
    multipleOf: int | None = None
    enum: list[int] | None = None
    title: str | None = None

    @classmethod
    def init(cls, schema: dict[str, Any]) -> "StrictJsonSchemaInteger":
        return cls(
            type="integer",
            minimum=schema.get("minimum"),
            maximum=schema.get("maximum"),
            exclusiveMinimum=schema.get("exclusiveMinimum"),
            exclusiveMaximum=schema.get("exclusiveMaximum"),
            multipleOf=schema.get("multipleOf"),
            enum=schema.get("enum"),
            title=schema.get("title"),
        )


class StrictJsonSchemaBoolean(BaseModel):
    type: Literal["boolean"]
    title: str | None = None

    @classmethod
    def init(cls, schema: dict[str, Any]) -> "StrictJsonSchemaBoolean":
        return cls(type="boolean", title=schema.get("title"))


class StrictJsonSchemaNull(BaseModel):
    type: Literal["null"]
    title: str | None = None

    @classmethod
    def init(cls, schema: dict[str, Any]) -> "StrictJsonSchemaNull":
        return cls(type="null", title=schema.get("title"))


class StrictJsonSchemaAnyOf(BaseModel):
    anyOf: list["StrictJsonSchema"]
    title: str | None = None

    @classmethod
    def init(cls, schema: dict[str, Any]) -> "StrictJsonSchemaAnyOf":
        raw_members = schema.get("anyOf") or []
        members = [init_strict_json_schema(s) for s in raw_members]
        return cls(anyOf=members, title=schema.get("title"))


class StrictJsonSchemaEnum(BaseModel):
    enum: list[str | int | float | bool | None]
    title: str | None = None

    @classmethod
    def init(cls, schema: dict[str, Any]) -> "StrictJsonSchemaEnum":
        return cls(enum=list(schema.get("enum") or []), title=schema.get("title"))


StrictJsonSchema = (
    StrictJsonSchemaObject
    | StrictJsonSchemaArray
    | StrictJsonSchemaString
    | StrictJsonSchemaNumber
    | StrictJsonSchemaInteger
    | StrictJsonSchemaBoolean
    | StrictJsonSchemaNull
    | StrictJsonSchemaAnyOf
    | StrictJsonSchemaEnum
)


def init_strict_json_schema(schema: dict[str, Any]) -> StrictJsonSchema:
    """Construct a strict schema node from a possibly non-strict JSON Schema dict."""
    if "anyOf" in schema:
        return StrictJsonSchemaAnyOf.init(schema)

    # If no explicit type but enum present, keep as enum-only schema
    if "type" not in schema:
        if "enum" in schema:
            return StrictJsonSchemaEnum.init(schema)
        raise ValueError("Schema must include a type or an anyOf/enum")

    schema_type = schema["type"]
    if schema_type == "object":
        return StrictJsonSchemaObject.init(schema)
    if schema_type == "array":
        return StrictJsonSchemaArray.init(schema)
    if schema_type == "string":
        return StrictJsonSchemaString.init(schema)
    if schema_type == "number":
        return StrictJsonSchemaNumber.init(schema)
    if schema_type == "integer":
        return StrictJsonSchemaInteger.init(schema)
    if schema_type == "boolean":
        return StrictJsonSchemaBoolean.init(schema)
    if schema_type == "null":
        return StrictJsonSchemaNull.init(schema)

    raise ValueError(f"Unsupported schema type: {schema_type}")

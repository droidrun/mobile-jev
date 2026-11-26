from collections.abc import Mapping
from datetime import datetime, tzinfo
from typing import Annotated, Literal, Optional, Union
from uuid import UUID

import pytest
from llama_index.core.tools import ToolSelection
from mobilerun.core.devices import Device
from mobilerun.core.devices.recording import RecordingDevice
from mobilerun.core.extensions import Extension, action
from mobilerun.core.state.base import StateProvider
from mobilerun.core.state.ui_state import UIState
from mobilerun.errors.operators import ExtensionActionError
from pydantic import (
    AfterValidator,
    AliasChoices,
    AliasPath,
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    field_validator,
)
from pydantic_core import PydanticCustomError, SchemaValidator, core_schema
from workflows import Context

from mobilerun_agents.agent.logs.redaction import Redactor
from mobilerun_agents.agent.tools.base import Source, tool
from mobilerun_agents.agent.tools.helpers import visual_tool_names
from mobilerun_agents.agent.tools.helpers.errors import tool_error_summary
from mobilerun_agents.agent.tools.registry import build_registry
from mobilerun_agents.agent.workflows.core.state import AgentState
from mobilerun_agents.config.user_config import ToolsConfig
from tests.unit.agent.action_selector.fixtures import Turn
from tests.unit.device_fixtures import MemoryDevice


def call(tool_name, **args):
    return ToolSelection(tool_id="x", tool_name=tool_name, tool_kwargs=args)


@pytest.mark.anyio
async def test_resources_are_injected_without_exposing_them_as_arguments():
    @tool(sensitive_parameters=frozenset({"label"}))
    async def mark(label: str, workflow: Context[AgentState], phone: Device) -> str:
        """Label this device."""
        return f"{phone.platform}: {(await workflow.store.get_state()).goal}: {label}"

    turn = Turn(goal="bench")
    registry = build_registry(turn.device, [mark], ToolsConfig())
    schema = registry.by_name["mark"].metadata.get_parameters_dict()["properties"]
    assert set(schema) == {"label"}
    assert (
        await registry.execute(call("mark", label="sample name"), turn.ctx)
        == "android: bench: sample name"
    )
    assert registry.protected_values == {"sample name"}
    assert registry.redactor.text("sample%20name") == "***"
    with pytest.raises(ExtensionActionError):
        await registry.execute(call("mark", label="sample", phone="pretender"), turn.ctx)


@pytest.mark.anyio
async def test_native_action_preserves_outer_recording_and_hides_backend_exception_text():
    class Phone(MemoryDevice):
        async def open_url(self, url: str) -> None:
            raise RuntimeError("fixture refused the URL")

    phone = Phone(supported_operations=frozenset({"input_text", "open_url"}))
    outer = RecordingDevice(phone)
    turn = Turn(device=outer)
    result = await turn.registry.execute(call("input_text", text="ordinary text"), turn.ctx)
    assert result.success and phone.inputs == ["ordinary text"]
    assert [a["text"] for a in outer.recorded_actions] == ["ordinary text"]
    result = await turn.registry.execute(call("open_url", url="https://example.invalid"), turn.ctx)
    assert not result.success
    assert result.summary == "RuntimeError"


def test_registry_validates_tool_selection_against_dispatch_schema():
    turn = Turn()
    registry = turn.registry
    assert registry.validate(call("complete", success=True, message=None)) is None
    assert registry.validate(call("complete", success="true", message=None)) == (
        "Invalid arguments for complete: success: Input should be a valid boolean"
    )
    assert registry.validate(call("wait", unexpected=1)) == "Unexpected arguments for wait."
    assert registry.validate(call("unknown")) == "Unavailable tool: unknown"


@pytest.mark.anyio
async def test_tap_element_uses_snapshot():
    class Phone(MemoryDevice):
        async def tap(self, x: int, y: int) -> None:
            self.hit = (x, y)

    class Provider(StateProvider):
        capabilities = {"element_index"}

        async def get_state(self):
            return self.observation

    def ui(bounds):
        return UIState(
            [
                {
                    "index": 7,
                    "className": "android.widget.TextView",
                    "text": "Settings",
                    "bounds": bounds,
                }
            ],
            "screen",
            "",
            {},
            100,
            100,
        )

    phone = Phone(supported_operations=frozenset({"tap"}))
    provider = Provider(phone)
    phone._set_state_provider(provider)
    pinned = provider.observation = ui("0,0,20,20")
    turn = Turn(device=phone, ui=pinned)
    provider.observation = ui("70,70,90,90")
    assert await phone.state.refresh() is not pinned
    assert visual_tool_names(turn.registry.tools()) == ({"tap"}, {"tap_element"})
    result = await turn.registry.execute(call("tap_element", index=7), turn.ctx)
    assert result.success and phone.hit == (10, 10)
    assert result.summary == 'tapped [7] TextView "Settings"'


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("point", "refresh_in_action", "summary"),
    [
        ((12, 12), True, 'tapped (12, 12) Button "Open ZIP sheet" · focused: EditText "ZIP"'),
        ((30, 30), False, "tapped (30, 30)"),
    ],
)
async def test_raw_tap_summary_and_already_fetched_focus(point, refresh_in_action, summary):
    class Phone(MemoryDevice):
        async def tap(self, x: int, y: int) -> None:
            if refresh_in_action:
                await self.state.refresh()

    class Provider(StateProvider):
        reads = 0

        async def get_state(self):
            self.reads += 1
            return UIState(
                [],
                "",
                "",
                {
                    "focusedElement": {
                        "className": "android.widget.EditText",
                        "text": "private",
                        "hint": "ZIP",
                    }
                },
                100,
                100,
            )

    phone = Phone(supported_operations=frozenset({"tap"}))
    provider = Provider(phone)
    phone._set_state_provider(provider)
    await phone.state.refresh()  # Stale focused field must not leak into the result.
    pinned = UIState(
        [
            {
                "index": 7,
                "children": [
                    {
                        "index": 8,
                        "className": "android.widget.Button",
                        "bounds": "10,10,20,20",
                        "details": {"description": "Open ZIP sheet"},
                    }
                ],
            }
        ],
        "",
        "",
        {},
        100,
        100,
        use_normalized=True,  # Registry taps already contain native pixels.
    )
    turn = Turn(device=phone, ui=pinned)
    result = await turn.registry.execute(call("tap", x=point[0], y=point[1]), turn.ctx)
    assert result.success
    assert result.summary == summary
    assert provider.reads == 1 + refresh_in_action


@pytest.mark.anyio
async def test_base_none_result_has_no_none_suffix():
    class Phone(MemoryDevice):
        async def press_button(self, button: str) -> None:
            self.pressed = button

    phone = Phone(supported_operations=frozenset({"press_button"}))
    turn = Turn(device=phone)
    result = await turn.registry.execute(call("press_button", button="back"), turn.ctx)
    assert result.success and phone.pressed == "back"
    assert "None" not in result.summary


@pytest.mark.anyio
async def test_indexed_run_tools_describe_the_element_without_echoing_the_input():
    @tool
    async def long_press_element(index: int) -> None:
        """Long-press a visible element."""

    @tool(sensitive_parameters=frozenset({"text"}))
    async def input_text_on_element(index: int, text: str) -> str:
        """Type into a visible element."""
        return "saved"

    secret = "fixture-sensitive-value-" * 4
    turn = Turn(
        elements=[
            {"index": 2, "text": "x" * 80, "className": "android.widget.TextView"},
            {"index": 3, "text": secret, "className": "android.widget.EditText"},
        ]
    )
    registry = build_registry(
        turn.device, [long_press_element, input_text_on_element], ToolsConfig()
    )
    pressed = await registry.execute(call("long_press_element", index=2), turn.ctx)
    assert pressed.summary == 'long-pressed [2] TextView "' + "x" * 59 + '…"'
    typed = await registry.execute(call("input_text_on_element", index=3, text=secret), turn.ctx)
    assert typed.summary == "typed into [3] EditText → saved"


@pytest.mark.anyio
async def test_extension_names_and_exclusions_preserve_existing_routes():
    class Bench(Extension):
        default_name = "bench"

        @action
        async def ping(self) -> str:
            """Read bench status."""
            return "ready"

    @tool
    async def bench__ping() -> None:
        """Ping the bench."""

    phone = MemoryDevice(supported_operations=frozenset({"input_text"}))
    extension = await Bench().bind(phone)
    try:
        turn = Turn(device=phone)
        registry = turn.registry
        assert registry.by_name["bench__ping"].metadata.source == "extension:bench"
        assert registry.by_name["input_text"].metadata.source is Source.DEVICE
        assert "input_text" not in {
            m.metadata.get_name() for m in registry.tools(exclude={"input_text"})
        }
        result = await registry.execute(call("bench__ping"), turn.ctx)
        assert result.success and "ready" in result.summary
        with pytest.raises(ValueError, match="Duplicate tool name: bench__ping"):
            build_registry(phone, [bench__ping], ToolsConfig(exclude=["bench__ping"]))
    finally:
        await extension.unbind()


@pytest.mark.anyio
async def test_resolve_retains_native_extension_identity_and_arguments():
    class Vault(Extension):
        default_name = "vault"

        @action
        async def type_field(self, name: str, field: str) -> None:
            """Type one named field."""

    phone = MemoryDevice()
    vault = await Vault().bind(phone)
    try:
        turn = Turn(device=phone)
        assert turn.registry.resolve(
            call("vault__type_field", name="work", field="email")
        ) == ToolSelection(
            tool_id="x",
            tool_name="vault.type_field",
            tool_kwargs={"name": "work", "field": "email"},
        )
    finally:
        await vault.unbind()


class Address(BaseModel):
    model_config = ConfigDict(extra="forbid")
    city: str


class Parcel(BaseModel):
    address: Optional[Annotated[Address, AfterValidator(lambda address: address)]] = None
    value: int | str = 0
    stops: Union[tuple[Address, int], list[str]]
    tags: Mapping[str, int]


def test_only_open_mapping_keys_and_forbidden_extras_are_protected():
    with pytest.raises(ValidationError) as caught:
        Parcel.model_validate(
            {
                "address": {"city": 5, "fixture-extra": True},
                "value": [1],
                "stops": [{"city": 3}, 7],
                "tags": {"k": "unknown"},
            }
        )
    protected: list[str] = []
    summary = tool_error_summary(caught.value, Parcel, protected.append)
    assert "address.city:" in summary
    assert "stops.0.city:" in summary
    assert "tags.*:" in summary
    assert "fixture-extra" not in summary
    assert protected == ["fixture-extra", "k"]


@pytest.mark.parametrize(
    "field,key,nested,path",
    [
        (Field(validation_alias=AliasChoices("innerA", "innerB")), "innerB", False, "innerB"),
        (Field(validation_alias=AliasPath("items", 0)), "items", True, "items.0"),
        (
            Field(validation_alias=AliasChoices("innerA", AliasPath("items", 0))),
            "items",
            True,
            "items.0",
        ),
        (Field(alias="innerB"), "innerB", False, "innerB"),
    ],
)
def test_extra_key_is_protected_behind_validation_aliases(field, key, nested, path):
    class Outer(BaseModel):
        inner: Address = field

    class Output(BaseModel):
        outer: Outer

    value = {"city": "Berlin", "fixture-extra": True}
    with pytest.raises(ValidationError) as caught:
        Output.model_validate({"outer": {key: [value] if nested else value}})
    protected: list[str] = []
    summary = tool_error_summary(caught.value, Output, protected.append)
    assert protected == ["fixture-extra"]
    assert summary == f"outer.{path}.*: Extra inputs are not permitted"


def test_nonterminal_unknown_location_is_not_registered_as_an_extra_key():
    class Output(BaseModel):
        @field_validator("child")
        @classmethod
        def validate_child(cls, child: Address) -> Address:
            raise ValidationError.from_exception_data(
                "child",
                [{"type": "extra_forbidden", "loc": ("unknown", "fixture-extra"), "input": True}],
            )

        child: Address

    with pytest.raises(ValidationError) as caught:
        Output.model_validate({"child": {"city": "Berlin"}})
    protected: list[str] = []
    tool_error_summary(caught.value, Output, protected.append)
    assert protected == []


def test_model_and_function_union_tags_do_not_poison_the_redactor():
    class Output(BaseModel):
        target: Annotated[Address, AfterValidator(lambda address: address)] | list[Address]
        value: int | str

    with pytest.raises(ValidationError) as caught:
        Output.model_validate({"target": {"city": 3}, "value": [1]})
    redactor = Redactor()
    assert "target.city:" in tool_error_summary(caught.value, Output, redactor.add)
    assert redactor.revision == 0
    assert redactor.text("print(str(int_value))") == "print(str(int_value))"


def test_missing_schema_does_not_register_unknown_path_components():
    class Cart(BaseModel):
        items: list[Address]

    with pytest.raises(ValidationError) as caught:
        Cart.model_validate({"items": [{"city": 3}]})
    redactor = Redactor()
    tool_error_summary(caught.value, protect=redactor.add)
    assert redactor.revision == 0


class Primary(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["a"]
    count: int


class Secondary(BaseModel):
    kind: Literal["b"]
    fixture_extra: str = ""


class Tagged(BaseModel):
    choice: Primary | Secondary = Field(discriminator="kind")


def test_discriminated_union_keeps_fields_and_protects_extras():
    with pytest.raises(ValidationError) as caught:
        Tagged.model_validate({"choice": {"kind": "a", "count": "bad", "fixture_extra": 1}})
    protected: list[str] = []
    summary = tool_error_summary(caught.value, Tagged, protected.append)
    assert "choice.count:" in summary
    assert "fixture_extra" not in summary
    assert protected == ["fixture_extra"]


class BrokenOffset(tzinfo):
    def utcoffset(self, dt):
        raise ValueError("fixture-echo")


class BrokenAttribute:
    @property
    def amount(self):
        raise ValueError("fixture-echo")


class AttributeModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    amount: int


class AssignmentModel(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
    amount: int = 0


def assign_unknown_attribute(name):
    setattr(AssignmentModel(), name, 1)


class BrokenMapping(Mapping):
    def __getitem__(self, key):
        raise KeyError(key)

    def __iter__(self):
        raise ValueError("fixture-echo")

    def __len__(self):
        return 1


def broken_iterable():
    yield 1
    raise ValueError("fixture-echo")


def test_custom_validation_message_does_not_echo_the_input():
    class Output(BaseModel):
        code: str

        @field_validator("code")
        @classmethod
        def validate_code(cls, value: str) -> str:
            raise PydanticCustomError("custom_code", "Bad code {value}", {"value": value})

    with pytest.raises(ValidationError) as caught:
        Output.model_validate({"code": "fixture-secret"})
    [issue] = caught.value.errors(include_input=False, include_context=False)
    assert issue["msg"] == "Bad code fixture-secret"
    assert tool_error_summary(caught.value, Output) == "code: Invalid value"


@pytest.mark.parametrize(
    ("validate", "value", "error_type", "echo"),
    [
        (
            Tagged.model_validate,
            {"choice": {"kind": "fixture-echo"}},
            "union_tag_invalid",
            "fixture-echo",
        ),
        (AttributeModel.model_validate, BrokenAttribute(), "get_attribute_error", "fixture-echo"),
        (assign_unknown_attribute, "fixture-echo", "no_such_attribute", "fixture-echo"),
        (
            TypeAdapter(dict[str, int]).validate_python,
            BrokenMapping(),
            "mapping_type",
            "fixture-echo",
        ),
        (
            TypeAdapter(list[int]).validate_python,
            broken_iterable(),
            "iteration_error",
            "fixture-echo",
        ),
        (TypeAdapter(UUID).validate_python, "x" + "0" * 35, "uuid_parsing", "`x`"),
        (
            TypeAdapter(bytes, config=ConfigDict(val_json_bytes="hex")).validate_json,
            '"z1"',
            "bytes_invalid_encoding",
            "'z'",
        ),
        (
            SchemaValidator(core_schema.datetime_schema(tz_constraint=0)).validate_python,
            "2020-01-01T00:00:00+02:34",
            "timezone_offset",
            "9240",
        ),
        (
            TypeAdapter(AwareDatetime).validate_python,
            datetime(2020, 1, 1, tzinfo=BrokenOffset()),
            "datetime_object_invalid",
            "fixture-echo",
        ),
    ],
)
def test_summary_does_not_echo_input_or_input_originated_exceptions(
    validate, value, error_type, echo
):
    with pytest.raises(ValidationError) as caught:
        validate(value)
    [issue] = caught.value.errors(include_input=False, include_context=False)
    assert issue["type"] == error_type
    assert echo in issue["msg"]
    assert echo not in tool_error_summary(caught.value)

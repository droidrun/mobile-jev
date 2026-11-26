from mobilerun.core.devices import Device
from workflows import Context

from mobilerun_agents.agent.tools.base import tool


def test_tool_schema_hides_resources_by_type_not_parameter_name():
    @tool
    async def label(text: str, phone: Device, workflow: Context) -> str:
        """Label the connected phone."""
        return text

    assert set(label.compile().input_schema["properties"]) == {"text"}

"""
Simple workflow to open an app based on a description.
"""

import asyncio
import json
import logging
from typing import Unpack

from llama_index.core.llms.llm import LLM
from mobilerun.core.devices import AndroidDevice, Device
from mobilerun.logs.handlers import VERBOSE, log_verbose
from workflows import Context, Workflow, step
from workflows.events import StartEvent, StopEvent

from mobilerun.agents.llm.inference import acomplete_with_retries
from mobilerun.agents.llm.llm_picker import load_llm
from mobilerun.agents.workflows import WorkflowOptionsWithoutTimeout

logger = logging.getLogger("mobilerun")


class AppStarter(Workflow):
    """
    A simple workflow that opens an app based on a description.

    The workflow uses an LLM to intelligently match the app description
    to an installed app's package name, then opens it.
    """

    def __init__(
        self,
        driver: Device,
        llm: LLM,
        timeout: int = 60,
        stream: bool = False,
        **kwargs: Unpack[WorkflowOptionsWithoutTimeout],
    ) -> None:
        """
        Initialize the OpenAppWorkflow.

        Args:
            driver: Framework device with list_apps()/start_app() methods
            llm: An LLM instance (e.g., OpenAI) to determine which app to open
            timeout: Workflow timeout in seconds (default: 60)
            stream: If True, stream LLM response to console in real-time
            **kwargs: Additional arguments passed to Workflow
        """
        super().__init__(timeout=timeout, **kwargs)
        self.driver = driver
        self.llm = llm
        self.stream = stream

    @step
    async def open_app_step(self, ev: StartEvent, ctx: Context) -> StopEvent:
        """
        Opens an app based on the provided description.

        Expected StartEvent attributes:
            - app_description (str): The name or description of the app to open

        Returns:
            StopEvent with the result of the open_app operation
        """
        app_description = ev.app_description

        # Get list of installed apps
        apps = await self.driver.list_apps(include_system=True)

        # Format apps list for LLM
        apps_list = "\n".join(
            [
                f"- {app['label']} (package: {app['package_name'] if 'package_name' in app else app['package']})"
                for app in apps
            ]
        )

        # Construct prompt for LLM
        prompt = f"""Given the following list of installed apps and a user's description, determine which app package name to open.

Installed Apps:
{apps_list}

User's Request: "{app_description}"

Return ONLY a JSON object with the following structure:
{{
    "package": "com.example.package"
}}

If no app matches the user's request, return:
{{
    "package": null
}}

Choose the most appropriate app based on the description. Return the package name of the best match, or null if no app matches."""

        # Get LLM response
        log_verbose(logger, " AppOpener response:", color="blue")
        response = await acomplete_with_retries(self.llm, prompt, stream=self.stream)
        response_text = response.text.strip()

        # Parse JSON response - extract content between { and }
        try:
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            json_str = response_text[start:end]
            result_json = json.loads(json_str)
            package_name = result_json["package"]
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            return StopEvent(result=f"Error parsing LLM response: {e}. Response: {response_text}")

        if not package_name:
            logger.warning(f"No matching app found for: {app_description}")
            return StopEvent(
                result=f"Could not open app: no installed app matches '{app_description}'"
            )

        logger.log(VERBOSE, f"Starting app {package_name}")
        result = await self.driver.actions.call(
            "device.start_app", {"package": package_name}, audience="operator"
        )

        return StopEvent(result=result)


# Example usage
async def main() -> None:
    """
    Example of how to use the OpenAppWorkflow.
    """
    # Initialize driver with device serial (None for default device)
    driver = AndroidDevice(serial=None)
    await driver.connect()

    # Initialize LLM
    llm = load_llm("OpenAIResponses", model="gpt-6.1-sol")

    # Create workflow instance
    workflow = AppStarter(driver=driver, llm=llm, timeout=60, verbose=True)

    # Run workflow to open an app
    result = await workflow.run(app_description="Settings")

    logger.log(VERBOSE, f"Result: {result}")


if __name__ == "__main__":
    asyncio.run(main())

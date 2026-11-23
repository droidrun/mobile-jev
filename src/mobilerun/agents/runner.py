"""Agent execution shared by the CLI and programmatic callers."""

import logging
import os  # Third-party libraries consume these process environment switches.
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from mobilerun.clients.portal.http import is_portal_http_url
from mobilerun.core.credentials import FileCredentials
from mobilerun.errors import CloudAuthRequired, CloudTaskTimeout
from mobilerun.logs.actions import action_counting
from mobilerun.logs.handlers import LogLevel, apply_log_level
from mobilerun_core_local.driver.ios import validate_ios_portal_url
from mobilerun_core_local.driver.visual_remote import (
    VISUAL_REMOTE_CONNECTION,
)

from mobilerun.agents.clients.portal.keyboard import disable_portal_keyboard
from mobilerun.agents.config.loader import ConfigLoader
from mobilerun.agents.config.user_config import MobileConfig
from mobilerun.agents.errors import KeyboardCleanupFailed
from mobilerun.agents.llm.llm_picker import load_llm
from mobilerun.agents.logs.error_descriptions import describe_error
from mobilerun.agents.logs.events import EventHandler
from mobilerun.agents.workflows.droid import MobileAgent
from mobilerun.agents.workflows.droid.events import ResultEvent


@dataclass
class AgentRunResult:
    success: bool
    cloud_result: Any = None
    agent_result: ResultEvent | None = None
    error: str | None = None


def _force_screenshot_only_vision(config: MobileConfig) -> None:
    config.agent.vision_only = True
    config.agent.manager.vision = True
    config.agent.executor.vision = True
    config.agent.fast_agent.vision = True


async def _cleanup_portal_keyboard(
    config: MobileConfig,
    serial: str | None = None,
    failure_handler: Callable[[str], None] | None = None,
) -> None:
    """Disable the keyboard when this run could have enabled it."""
    platform = (config.device.platform or "").lower()
    backend = (config.device.control_backend or "").lower()
    serial = serial or config.device.serial
    if (
        platform == "ios"
        or backend == VISUAL_REMOTE_CONNECTION
        or not serial
        or is_portal_http_url(serial)
    ):
        return
    try:
        await disable_portal_keyboard(serial)
    except KeyboardCleanupFailed:
        message = "Failed to disable Mobilerun keyboard"
        if failure_handler:
            failure_handler(message)
        else:
            logging.getLogger(__name__).warning(message)


async def run_agent(
    command: str,
    config_path: str | None = None,
    device: Any = None,
    target: str | None = None,
    device_kind: str | None = None,
    device_base_url: str | None = None,
    agent: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    steps: int | None = None,
    base_url: str | None = None,
    api_base: str | None = None,
    vision: bool | None = None,
    vision_only: bool | None = None,
    manager_vision: bool | None = None,
    executor_vision: bool | None = None,
    fast_agent_vision: bool | None = None,
    reasoning: bool | None = None,
    stream: bool | None = None,
    tracing: bool | None = None,
    debug: bool | None = None,
    log_level: LogLevel | None = None,
    tcp: bool | None = None,
    control_backend: str | None = None,
    device_id: str | None = None,
    save_trajectory: str | None = None,
    ios: bool = False,
    temperature: float | None = None,
    keyboard_cleanup_failure_handler: Callable[[str], None] | None = None,
    on_config: Callable[[MobileConfig, bool], None] | None = None,
    on_cloud_warning: Callable[[str], None] | None = None,
    on_cloud_task_started: Callable[[str], None] | None = None,
    **kwargs: Any,
) -> AgentRunResult:
    """Run a command and return its outcome without rendering terminal output."""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "false"
    config = ConfigLoader.load(None if device_kind == "cloud" else config_path)
    level = apply_log_level(config.logging, log_level, debug)
    debug_mode = level == "debug"
    if on_config:
        on_config(config, debug_mode)
    if device_kind == "cloud":
        ignored = []
        for option, value in (
            ("--config (including configured local LLM settings)", config_path),
            ("--agent", agent),
            ("--provider", provider),
            ("--model", model),
            ("--temperature", temperature),
            ("--base_url", base_url),
            ("--api_base", api_base),
            ("--vision", vision),
            ("--vision-only", vision_only),
            ("--reasoning", reasoning),
            ("--stream", stream),
            ("--tracing", tracing),
            ("--save-trajectory", save_trajectory),
        ):
            if value is not None:
                ignored.append(option)
        if ignored and on_cloud_warning:
            on_cloud_warning("Warning: cloud runs ignore local-only options: " + ", ".join(ignored))
        try:
            cloud_device = device
            if cloud_device is None:
                raise ValueError("A connected cloud device is required.")
            if on_cloud_task_started:
                cloud_device._task_started_callback = on_cloud_task_started
            cloud_result = await cloud_device.do(command, max_steps=steps)
        except (CloudAuthRequired, CloudTaskTimeout) as exc:
            return AgentRunResult(success=False, error=str(exc))
        return AgentRunResult(success=cloud_result.success, cloud_result=cloud_result)

    logger = logging.getLogger("mobilerun")
    droid_agent = None

    try:
        logger.info(f"Goal: {command}")

        # ================================================================
        # STEP 1: Apply CLI overrides via direct mutation
        # ================================================================

        # Vision overrides
        if vision_only is not None:
            config.agent.vision_only = vision_only

        if config.agent.vision_only:
            _force_screenshot_only_vision(config)
            logger.debug("CLI override: vision_only=True")
        elif vision is not None:
            # --vision flag overrides all agents
            config.agent.manager.vision = vision
            config.agent.executor.vision = vision
            config.agent.fast_agent.vision = vision
            logger.debug(f"CLI override: vision={vision} (all agents)")
        else:
            # Apply individual agent vision overrides
            if manager_vision is not None:
                config.agent.manager.vision = manager_vision
            if executor_vision is not None:
                config.agent.executor.vision = executor_vision
            if fast_agent_vision is not None:
                config.agent.fast_agent.vision = fast_agent_vision

        # Agent overrides
        if agent is not None:
            config.agent.name = agent
        if steps is not None:
            config.agent.max_steps = steps
        if reasoning is not None:
            config.agent.reasoning = reasoning
        if stream is not None:
            config.agent.streaming = stream

        # Explicit CLI target overrides only device fields; config may still
        # provide agent, provider, and other runtime settings.
        if target is not None:
            config.device.control_backend = None
            if device_kind == "android":
                config.device.platform = "android"
                config.device.serial = target
            elif device_kind == "ios":
                config.device.platform = "ios"
                config.device.serial = validate_ios_portal_url(target)
            elif device_kind == "cloud":
                config.device.serial = None
                config.device.device_id = target
                # Prevent the Android agent from interpreting a cloud target
                # as permission to auto-discover a local ADB device.
                config.device.control_backend = "cloud"
            elif device_kind == "visual_remote":
                config.device.platform = "android"
                config.device.serial = target
                config.device.control_backend = VISUAL_REMOTE_CONNECTION
        if tcp is not None:
            config.device.use_tcp = tcp
        if control_backend is not None:
            config.device.control_backend = control_backend
        if device_id is not None:
            config.device.device_id = device_id

        if (config.device.control_backend or "").strip().lower() == VISUAL_REMOTE_CONNECTION:
            _force_screenshot_only_vision(config)
            logger.debug("Control backend visual-remote forces screenshot-only vision")

        # Logging overrides
        if save_trajectory is not None:
            config.logging.save_trajectory = save_trajectory

        # Tracing overrides
        if tracing is not None:
            config.tracing.enabled = tracing

        # Legacy internal iOS callers no longer discover devices automatically.
        if ios and target is None:
            config.device.platform = "ios"
            if (config.device.control_backend or "").lower() == VISUAL_REMOTE_CONNECTION:
                pass
            elif config.device.serial:
                config.device.serial = validate_ios_portal_url(config.device.serial)
            else:
                raise ValueError("An explicit iOS URL is required.")

        # ================================================================
        # STEP 2: Initialize MobileAgent with config
        # ================================================================

        mode = "planning with reasoning" if config.agent.reasoning else "direct execution"
        logger.debug(f"Agent mode: {mode}")
        logger.debug(
            f"Vision settings: Manager={config.agent.manager.vision}, "
            f"Executor={config.agent.executor.vision}, FastAgent={config.agent.fast_agent.vision}"
        )

        if config.tracing.enabled:
            logger.debug("Tracing enabled")

        # Build MobileAgent kwargs for LLM loading
        droid_agent_kwargs: dict[str, Any] = {"runtype": "cli"}
        llm = None

        if provider or model:
            assert (
                provider and model
            ), "Either both provider and model must be provided or none of them"
            llm_kwargs: dict[str, Any] = {}
            if temperature is not None:
                llm_kwargs["temperature"] = temperature
            if base_url is not None:
                llm_kwargs["base_url"] = base_url
            if api_base is not None:
                llm_kwargs["api_base"] = api_base
            llm = load_llm(provider, model=model, **llm_kwargs, **kwargs)
        else:
            if temperature is not None:
                droid_agent_kwargs["temperature"] = temperature
            if base_url is not None:
                droid_agent_kwargs["base_url"] = base_url
            if api_base is not None:
                droid_agent_kwargs["api_base"] = api_base

        if device is not None:
            from mobilerun.core.devices.cloud import CloudDevice

            if config.credentials.enabled and not isinstance(device, CloudDevice):
                await FileCredentials(config.credentials.file_path).bind(device)
            droid_agent_kwargs["driver"] = device
        elif config.credentials.enabled:
            raise ValueError("A connected device is required when credentials are enabled.")

        droid_agent = MobileAgent(
            goal=command,
            llms=llm,
            config=config,
            timeout=1000,
            **droid_agent_kwargs,
        )

        # ================================================================
        # STEP 3: Run agent
        # ================================================================

        logger.debug("▶  Starting agent execution...")
        logger.debug("Press Ctrl+C to stop")

        try:
            with action_counting() as counter:
                event_handler = EventHandler(counter)
                handler: Any = droid_agent.run()
                async for event in handler.stream_events():
                    event_handler.handle(event)
                result: ResultEvent = await handler
                return AgentRunResult(success=result.success, agent_result=result)

        except KeyboardInterrupt:
            logger.info("Stopped by user")
            return AgentRunResult(success=False)

        except Exception as e:
            err_desc = describe_error(e) or type(e).__name__
            logger.error(f" Error: {err_desc}")
            if config.logging.debug:
                logger.debug(traceback.format_exc())
            return AgentRunResult(success=False)

    except Exception as e:
        err_desc = describe_error(e) or type(e).__name__
        logger.error(f" Setup error: {err_desc}")
        if debug_mode:
            logger.debug(traceback.format_exc())
        return AgentRunResult(success=False)
    finally:
        serial = (
            droid_agent.android_serial if droid_agent is not None else None
        ) or config.device.serial
        await _cleanup_portal_keyboard(config, serial, keyboard_cleanup_failure_handler)

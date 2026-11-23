"""
MobileAgent - A wrapper class that coordinates the planning and execution of tasks
to achieve a user's goal on a mobile device.

Architecture:
- When reasoning=False: Uses FastAgent directly
- When reasoning=True: Uses Manager (planning) + Executor (action) workflows
"""

import asyncio
import logging
import time
import traceback
from typing import TYPE_CHECKING, Any, Awaitable, Type

from async_adbutils import adb
from llama_index.core.llms.llm import LLM
from llama_index.core.workflow import Context, StartEvent, StopEvent, Workflow, step
from mobilerun.clients.portal.http import (
    create_android_portal_http_driver,
    is_portal_http_url,
)
from mobilerun.clients.portal.setup import portal_version_kwargs
from mobilerun.config.user_config import MobileConfig as BaseMobileConfig
from mobilerun.core.devices import (
    AndroidDevice,
    Device,
    IOSDevice,
    RecordingDevice,
    RemoteDevice,
    StealthDevice,
)
from mobilerun.core.devices.components.wrapper import DriverDevice
from mobilerun.core.state.options import StateProviderOptions, android_state_provider
from mobilerun.core.state.screenshot import (
    ScreenshotOnlyStateProvider,
)
from mobilerun.errors import DeviceOperationNotSupported
from mobilerun.logs.handlers import log_verbose
from mobilerun.macro.recorder import MacroRecorder
from mobilerun_core_local.driver.android.portal import ensure_portal_ready
from mobilerun_core_local.driver.base import DeviceDisconnectedError
from mobilerun_core_local.driver.ios import (
    discover_ios_device,
)
from mobilerun_core_local.driver.visual_remote import (
    VISUAL_REMOTE_CONNECTION,
    VISUAL_REMOTE_DEFAULT_URL,
)
from opentelemetry import trace
from pydantic import BaseModel
from workflows.events import Event
from workflows.handler import WorkflowHandler

from mobilerun.agents.config.settings import get_agent_settings
from mobilerun.agents.config.user_config import (
    DEFAULT_DISABLED_TOOLS,
    AgentConfig,
    CredentialsConfig,
    DeviceConfig,
    LoggingConfig,
    MobileConfig,
    TelemetryConfig,
    ToolsConfig,
    TracingConfig,
)
from mobilerun.agents.context.context_provider import ContextProvider
from mobilerun.agents.llm.llm_loader import (
    load_agent_llms,
    merge_llms_with_config,
)
from mobilerun.agents.logs.error_descriptions import describe_error
from mobilerun.agents.mcp.adapter import mcp_to_mobilerun_tools
from mobilerun.agents.mcp.client import MCPClientManager
from mobilerun.agents.mcp.config import MCPConfig
from mobilerun.agents.telemetry import (
    MobileAgentFinalizeEvent,
    MobileAgentInitEvent,
    capture,
    flush,
)
from mobilerun.agents.telemetry.phoenix import clean_span
from mobilerun.agents.telemetry.setup import (
    record_langfuse_screenshot,
    session_context,
    setup_tracing,
)
from mobilerun.agents.tools.action_context import ActionContext
from mobilerun.agents.tools.signatures import build_tool_registry
from mobilerun.agents.trajectory import TrajectoryWriter
from mobilerun.agents.trajectory.trajectory import Trajectory
from mobilerun.agents.utils.prompt_resolver import PromptResolver
from mobilerun.agents.utils.vision_sizing import (
    VisionResizePolicy,
    model_uses_normalized_coordinates,
)
from mobilerun.agents.workflows.common.events import RecordUIStateEvent, ScreenshotEvent
from mobilerun.agents.workflows.droid.events import (
    ExecutorInputEvent,
    ExecutorResultEvent,
    ExternalUserMessageDroppedEvent,
    FastAgentExecuteEvent,
    FastAgentResultEvent,
    FinalizeEvent,
    ManagerInputEvent,
    ManagerPlanEvent,
    ResultEvent,
)
from mobilerun.agents.workflows.droid.state import MobileAgentState, QueuedUserMessage
from mobilerun.agents.workflows.executor import ExecutorAgent
from mobilerun.agents.workflows.external import list_agents, load_agent, run_with_context
from mobilerun.agents.workflows.fast_agent import FastAgent
from mobilerun.agents.workflows.fast_agent.events import FastAgentCompletion
from mobilerun.agents.workflows.manager import ManagerAgent, StatelessManagerAgent
from mobilerun.agents.workflows.manager.prompts import ManagerResponseValidationError
from mobilerun.agents.workflows.oneflows.structured_output_agent import (
    StructuredOutputAgent,
)

if TYPE_CHECKING:
    from mobilerun.core.state.base import StateProvider

logger = logging.getLogger("mobilerun")

_COORDINATE_TOOL_NAMES = {"click_at", "click_area", "long_press_at"}
_WORKFLOW_DEADLINE_KEY = "mobile_agent_workflow_deadline"
_ABANDONED_FINALIZE_TASKS: set[asyncio.Future[Any]] = set()


def _reap_abandoned_finalize_task(task: asyncio.Future[Any]) -> None:
    """Consume a task's terminal exception without waiting for it."""
    _ABANDONED_FINALIZE_TASKS.discard(task)
    try:
        task.result()
    except (asyncio.CancelledError, Exception):
        pass


async def _run_finalize_stage(
    name: str,
    awaitable: Awaitable[Any],
    deadline: float,
    cap: float,
    *,
    continue_in_background: bool = False,
) -> tuple[bool, Any]:
    """Run one best-effort epilog stage without letting cancellation linger."""

    async def background_operation() -> Any:
        try:
            async with asyncio.timeout(cap):
                return await awaitable
        except TimeoutError:
            logger.warning("Final %s background cleanup timed out", name)
            raise

    # Schedule the original awaitable directly. A wrapper cancelled before its
    # first instruction would leave a captured coroutine unawaited.
    task = asyncio.ensure_future(background_operation() if continue_in_background else awaitable)
    try:
        remaining = min(cap, deadline - time.monotonic())
        if remaining <= 0:
            logger.debug("Skipping final %s: epilog budget exhausted", name)
            return False, None
        done, _ = await asyncio.wait({task}, timeout=remaining)
        if not done:
            logger.warning("Final %s timed out", name)
            return False, None
        try:
            return True, task.result()
        except asyncio.CancelledError:
            # A best-effort child may cancel itself; only cancellation of this
            # helper's await is an external cancellation of finalize.
            logger.warning("Final %s was cancelled", name)
            return False, None
        except Exception as exc:
            logger.warning("Final %s failed: %s", name, exc)
            return False, None
    finally:
        if not task.done():
            if not continue_in_background:
                task.cancel()
            _ABANDONED_FINALIZE_TASKS.add(task)
            task.add_done_callback(_reap_abandoned_finalize_task)


def _structured_output_extraction(
    structured_agent: StructuredOutputAgent, ctx: Context
) -> Awaitable[StopEvent]:
    """Run direct extraction with the observations of the nested workflow path."""

    @clean_span("StructuredOutputAgent.extract_structured_output")
    async def extract() -> StopEvent:
        return await structured_agent.extract_structured_output(ctx, StartEvent())

    @clean_span("StructuredOutputAgent.run")
    async def run() -> StopEvent:
        return await extract()

    return run()


def _normalize_control_backend(control_backend: str | None) -> str | None:
    if control_backend is None:
        return None
    normalized = control_backend.strip().lower()
    return normalized or None


def _force_screenshot_only_vision(agent_config: AgentConfig) -> None:
    agent_config.vision_only = True
    agent_config.manager.vision = True
    agent_config.executor.vision = True
    agent_config.fast_agent.vision = True


def _effective_disabled_tools(
    disabled_tools: list[str],
    state_provider: "StateProvider",
    vision_enabled: bool = False,
    explicit: bool = False,
) -> list[str]:
    requires_coords = state_provider.requires_coordinate_tools
    if requires_coords:
        # Screenshot-only / visual-remote modes cannot operate without coordinate
        # tools. Strict supersets of the legacy v5 default (e.g. default +
        # wait) get warn-and-strip — migration can't remove those because the
        # extras are intentional. Everything else (exact default or custom
        # list) raises: the exact default should have been migrated to None.
        if explicit:
            disabled_set = set(disabled_tools)
            blocked = sorted(disabled_set & _COORDINATE_TOOL_NAMES)
            if blocked:
                if set(DEFAULT_DISABLED_TOOLS) < disabled_set:
                    logger.warning(
                        "Legacy disabled_tools list %s contains coordinate tools "
                        "that the active state provider requires; stripping them "
                        "to allow startup. Consider setting tools.disabled_tools "
                        "to null (framework default) and listing only the extras "
                        "you actually want disabled.",
                        disabled_tools,
                    )
                else:
                    raise ValueError(
                        f"Cannot disable coordinate tools {blocked} when the "
                        "state provider requires them (vision_only=True or "
                        "visual remote control_backend). Remove these from "
                        "tools.disabled_tools."
                    )
        return [name for name in disabled_tools if name not in _COORDINATE_TOOL_NAMES]
    # Auto-unmask click_at only when (a) the caller didn't supply an explicit
    # list, and (b) the provider maps model-visible coordinates to the
    # driver's tap input space (directly, via the vision coordinate
    # contract's convert_point scaling, or via normalized 0-1000 conversion —
    # both Android and iOS set the flag accordingly).
    coords_align = state_provider.screenshot_matches_input_coords
    if vision_enabled and not explicit and coords_align:
        return [name for name in disabled_tools if name != "click_at"]
    return disabled_tools


class MobileAgent(Workflow):
    """
    A wrapper class that coordinates between agents to achieve a user's goal.

    Reasoning modes:
    - reasoning=False: Uses FastAgent directly for immediate execution
    - reasoning=True: Uses ManagerAgent (planning) + ExecutorAgent (actions)
    """

    _tracing_session_id: str | None = None
    _tracing_user_id: str | None = None

    def __init__(
        self,
        goal: str,
        config: BaseMobileConfig | None = None,
        llms: dict[str, LLM] | LLM | None = None,
        custom_tools: dict | None = None,
        variables: dict | None = None,
        output_model: Type[BaseModel] | None = None,
        prompts: dict[str, str] | None = None,
        driver: "Device | None" = None,
        state_provider: "StateProvider | None" = None,
        timeout: int = 1000,
        context_provider: ContextProvider | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if "credentials" in kwargs:
            raise TypeError("Attach credentials to the device, not MobileAgent")
        if config is not None:
            config = MobileConfig.from_base(config)
        self.user_id = kwargs.pop("user_id", None)
        self.runtype = kwargs.pop("runtype", "developer")
        self.context_provider = context_provider
        self.shared_state = MobileAgentState(
            instruction=goal,
            err_to_manager_thresh=2,
            user_id=self.user_id,
            runtype=self.runtype,
        )
        self.output_model = output_model

        # Initialize prompt resolver for custom prompts
        self.prompt_resolver = PromptResolver(custom_prompts=prompts)

        # Store custom variables in shared state
        if variables:
            self.shared_state.custom_variables = variables

        self.resolved_device_config = config.device if config else DeviceConfig()

        self.config = MobileConfig(
            agent=config.agent if config else AgentConfig(),
            device=self.resolved_device_config,
            tools=config.tools if config else ToolsConfig(),
            logging=config.logging if config else LoggingConfig(),
            tracing=config.tracing if config else TracingConfig(),
            telemetry=config.telemetry if config else TelemetryConfig(),
            llm_profiles=config.llm_profiles if config else {},
            credentials=config.credentials if config else CredentialsConfig(),
            external_agents=config.external_agents if config else {},
            mcp=config.mcp if config else MCPConfig(),
        )
        self.shared_state.telemetry_config_enabled = self.config.telemetry.enabled
        control_backend = _normalize_control_backend(self.resolved_device_config.control_backend)
        if (
            self.config.agent.vision_only
            or control_backend == VISUAL_REMOTE_CONNECTION
            or (state_provider is not None and state_provider.requires_coordinate_tools)
        ):
            _force_screenshot_only_vision(self.config.agent)

        # These are populated in start_handler (unless injected via __init__)
        self._injected_driver = driver
        self._injected_state_provider = state_provider
        self.driver: Device | None = None
        # Android serial resolved for this run.
        self.android_serial: str | None = None
        self.registry = None
        self.action_ctx: ActionContext | None = None
        self.state_provider: StateProvider | None = None

        super().__init__(*args, timeout=timeout, **kwargs)

        self._tracing_session_id = self.config.tracing.langfuse_session_id
        self._tracing_user_id = self.config.tracing.langfuse_user_id
        setup_tracing(self.config.tracing, agent=self)

        # Check if using external agent - skip LLM loading
        _BUILTIN_AGENT_NAMES = {"mobilerun", "droidrun"}
        self._using_external_agent = self.config.agent.name not in _BUILTIN_AGENT_NAMES

        self._stream_screenshots = get_agent_settings().stream_screenshots

        self.timeout = timeout

        # Store user custom tools
        self.user_custom_tools = custom_tools or {}

        # Initialize MCP manager (connections made lazily in start_handler)
        self.mcp_manager: MCPClientManager | None = None

        # Only load LLMs for native Mobilerun agents
        if not self._using_external_agent:
            if llms is None:
                if config is None:
                    raise ValueError(
                        "Either 'llms' or 'config' must be provided. "
                        "If llms is not provided, config is required to load LLMs from profiles."
                    )

                logger.debug(" Loading LLMs from config (llms not provided)...")

                llms = load_agent_llms(config=self.config, output_model=output_model, **kwargs)
            if isinstance(llms, dict):
                llms = merge_llms_with_config(
                    self.config, llms, output_model=output_model, **kwargs
                )
            elif isinstance(llms, LLM):
                pass
            else:
                raise ValueError(f"Invalid LLM type: {type(llms)}")

            if isinstance(llms, dict):
                role_llms: dict[str, LLM] = llms
            else:
                role_llms = dict.fromkeys(
                    (
                        "manager",
                        "executor",
                        "fast_agent",
                        "app_opener",
                        "structured_output",
                    ),
                    llms,
                )
            self.manager_llm = role_llms.get("manager")
            self.executor_llm = role_llms.get("executor")
            self.fast_agent_llm = role_llms.get("fast_agent")
            self.app_opener_llm = role_llms.get("app_opener")
            self.structured_output_llm = role_llms.get("structured_output", self.fast_agent_llm)
        else:
            logger.debug(f" Using external agent: {self.config.agent.name}")
            self.manager_llm = None
            self.executor_llm = None
            self.fast_agent_llm = None
            self.app_opener_llm = None
            self.structured_output_llm = None

        self.trajectory: Trajectory | None = None
        self.trajectory_writer: TrajectoryWriter | None = None
        self.macro_recorder: MacroRecorder | None = None

        # Sub-agents are created in __init__ but wired up in start_handler
        if self._using_external_agent:
            self.manager_agent = None
            self.executor_agent = None
        elif self.config.agent.reasoning:
            if self.manager_llm is None or self.executor_llm is None:
                raise ValueError("Reasoning mode requires manager and executor LLMs")
            if self.config.agent.manager.stateless:
                ManagerClass = StatelessManagerAgent
            else:
                ManagerClass = ManagerAgent

            # Pass None for tools-related params — wired up in start_handler
            self.manager_agent = ManagerClass(
                llm=self.manager_llm,
                action_ctx=None,
                state_provider=None,
                save_trajectory=self.config.logging.save_trajectory,
                shared_state=self.shared_state,
                agent_config=self.config.agent,
                registry=None,
                output_model=self.output_model,
                prompt_resolver=self.prompt_resolver,
                tracing_config=self.config.tracing,
                context_provider=context_provider,
                timeout=self.timeout,
            )
            self.executor_agent = ExecutorAgent(
                llm=self.executor_llm,
                registry=None,
                action_ctx=None,
                shared_state=self.shared_state,
                agent_config=self.config.agent,
                prompt_resolver=self.prompt_resolver,
                timeout=self.timeout,
            )
        else:
            self.manager_agent = None
            self.executor_agent = None

        # Telemetry init event is fired in start_handler after registry is built.
        self._init_prompts = prompts  # stash for telemetry
        self._init_timeout = timeout

        logger.debug("MobileAgent initialized successfully.")

    def run(self, *args: Any, **kwargs: Any) -> WorkflowHandler:
        from mobilerun.logs.handlers import ensure_agent_logging

        ensure_agent_logging()
        # Workflow tasks inherit this context when run() schedules them.
        with session_context(self._tracing_session_id, self._tracing_user_id, agent=self):
            return super().run(*args, **kwargs)

    async def _initialize_workflow_deadline(self, ctx: Context) -> None:
        """Store a deadline owned by the runtime execution of this run."""
        runtime_timeout = self._timeout
        workflow_deadline = (
            time.monotonic() + runtime_timeout if runtime_timeout is not None else None
        )
        await ctx.store.set(_WORKFLOW_DEADLINE_KEY, workflow_deadline)

    async def _initialize_run_trajectory(self) -> None:
        """Create trajectory state and a writer owned by the current run."""
        if self._using_external_agent or self.config.logging.save_trajectory == "none":
            self.trajectory = None
            self.trajectory_writer = None
            self.macro_recorder = None
            return

        self.trajectory = Trajectory(
            goal=self.shared_state.instruction,
            base_path=self.config.logging.trajectory_path,
        )
        self.trajectory_writer = TrajectoryWriter(queue_size=300)
        self.macro_recorder = MacroRecorder()
        await self.trajectory_writer.start()

    # ========================================================================
    # start_handler — creates driver, registry, action_ctx
    # ========================================================================

    @step
    async def start_handler(
        self, ctx: Context, ev: StartEvent
    ) -> FastAgentExecuteEvent | ManagerInputEvent | FinalizeEvent:
        await self._initialize_workflow_deadline(ctx)
        ctx.write_event_to_stream(ev)
        await self._initialize_run_trajectory()

        # ── 0. External agent — early exit ────────────────────────────
        if self._using_external_agent:
            agent_name = self.config.agent.name

            # Load the agent module
            agent_module = load_agent(agent_name)
            if not agent_module:
                available = list_agents()
                if available:
                    agents_str = ", ".join(available)
                    raise ValueError(
                        f"Failed to load external agent '{agent_name}'.\n"
                        f"Available agents: {agents_str}"
                    )
                raise ValueError(
                    f"External agent '{agent_name}' not found.\n"
                    "No external agents are currently installed.\n"
                    "Run: mobilerun agent run --help  to see available agents."
                )

            # Resolve config — missing section is fine, agent may use DEFAULT_CONFIG or env vars
            agent_config = self.config.external_agents.get(agent_name) or {}
            final_config = {**agent_module["config"], **agent_config}

            # Resolve device serial and get raw AdbDevice
            if self._injected_driver is not None:
                injected = self._injected_driver._unwrap()
                if not isinstance(injected, AndroidDevice):
                    raise ValueError(
                        "External agents require an injected AndroidDevice backed by ADB."
                    )
                device_serial = injected._serial
            else:
                device_serial = self.resolved_device_config.serial
            if is_portal_http_url(device_serial):
                raise ValueError(
                    "External agents need an ADB device; a Portal HTTP URL is not supported."
                )
            if device_serial is None:
                raise ValueError(
                    "No device given. Pass a device serial, e.g. "
                    "AndroidDevice('emulator-5554') or `mobilerun agent run ... DEVICE`."
                )

            adb_device = await adb.device(serial=device_serial)
            self.android_serial = device_serial

            log_verbose(logger, f"Using external agent: {agent_name}")

            result = await run_with_context(
                agent_module["run"],
                self.context_provider,
                self.shared_state,
                device=adb_device,
                config=final_config,
                max_steps=self.config.agent.max_steps,
            )

            return FinalizeEvent(success=result["success"], reason=result["reason"])

        # An injected device may have a screenshot-only provider even when the
        # independent config describes the default accessibility backend.
        if (
            self._injected_driver is not None
            and self._injected_state_provider is None
            and not self.config.agent.vision_only
        ):
            provider = self._injected_driver._build_default_state_provider()
            if provider.requires_coordinate_tools:
                _force_screenshot_only_vision(self.config.agent)

        # ── 1. Create driver ──────────────────────────────────────────
        if self.config.agent.reasoning:
            # The executor emits the coordinate actions, so its vision flag
            # matters as much as the manager's for the coordinate contract.
            vision_enabled = (
                self.config.agent.vision_only
                or self.config.agent.manager.vision
                or self.config.agent.executor.vision
            )
        else:
            vision_enabled = self.config.agent.vision_only or self.config.agent.fast_agent.vision

        # Resolve the model-facing screenshot size once, from every vision LLM
        # that will receive a screenshot, so the declared coordinate space equals
        # what each model actually grounds on after any provider-side downsize.
        # Conservative: use the smallest effective size across all recipients.
        if self.config.agent.reasoning:
            vision_llms = [
                llm
                for use, llm in (
                    (
                        self.config.agent.vision_only or self.config.agent.manager.vision,
                        self.manager_llm,
                    ),
                    (
                        self.config.agent.vision_only or self.config.agent.executor.vision,
                        self.executor_llm,
                    ),
                )
                if use and llm is not None
            ]
        elif (
            self.config.agent.vision_only or self.config.agent.fast_agent.vision
        ) and self.fast_agent_llm is not None:
            vision_llms = [self.fast_agent_llm]
        else:
            vision_llms = []
        vision_resize_policy = VisionResizePolicy.from_llms(
            vision_llms, max_side_cap=self.config.agent.model_screenshot_max_side
        )
        # In reasoning mode the Executor only sees a screenshot when the Manager
        # also captured one (manager.vision=True), so require both before
        # exposing coordinate clicks.
        if self.config.agent.reasoning:
            active_action_vision = (
                self.config.agent.manager.vision and self.config.agent.executor.vision
            )
        else:
            active_action_vision = self.config.agent.fast_agent.vision
        action_agent_has_vision = self.config.agent.vision_only or active_action_vision
        # The agent that emits coordinate actions decides the coordinate space.
        # Its own vision flag counts: the Manager also captures screenshots for
        # trajectories and streaming.
        coordinate_llm = self.executor_llm if self.config.agent.reasoning else self.fast_agent_llm
        coordinate_agent_vision = self.config.agent.vision_only or (
            self.config.agent.executor.vision
            if self.config.agent.reasoning
            else self.config.agent.fast_agent.vision
        )
        use_normalized = self.config.agent.use_normalized_coordinates or (
            coordinate_agent_vision
            and coordinate_llm is not None
            and model_uses_normalized_coordinates(coordinate_llm.metadata.model_name)
        )

        is_ios = self.resolved_device_config.platform.lower() == "ios"
        control_backend = _normalize_control_backend(self.resolved_device_config.control_backend)
        if control_backend and control_backend != VISUAL_REMOTE_CONNECTION:
            raise ValueError(
                "Unsupported device control backend "
                f"'{self.resolved_device_config.control_backend}'. Supported: "
                f"{VISUAL_REMOTE_CONNECTION}. Omit control backend for the "
                "platform default."
            )
        is_visual_remote = control_backend == VISUAL_REMOTE_CONNECTION

        if self._injected_driver is not None:
            driver = self._injected_driver
        elif is_visual_remote:
            visual_remote_url = self.resolved_device_config.serial or VISUAL_REMOTE_DEFAULT_URL
            driver = RemoteDevice(
                url=visual_remote_url,
                device_id=self.resolved_device_config.device_id,
            )
            await driver.connect()
        elif is_ios:
            ios_url = self.resolved_device_config.serial
            if not ios_url:
                ios_url = await discover_ios_device()
            driver = await IOSDevice(
                ios_url,
                token=self.resolved_device_config.resolve_auth_token(),
            )
            await driver.connect()
        elif is_portal_http_url(self.resolved_device_config.serial):
            # Portal HTTP talks to the app directly: no ADB serial, auto-setup,
            # or keyboard cleanup.
            portal_url = self.resolved_device_config.serial
            if portal_url is None:
                raise ValueError("Portal HTTP requires a URL")
            raw_driver = create_android_portal_http_driver(
                portal_url,
                self.resolved_device_config.resolve_auth_token(),
            )
            driver = DriverDevice(raw_driver, state_provider_factory=android_state_provider)
            await driver.connect()
        else:
            device_serial = self.resolved_device_config.serial
            if device_serial is None:
                raise ValueError(
                    "No device given. Pass a device serial, e.g. "
                    "AndroidDevice('emulator-5554') or `mobilerun agent run ... DEVICE`."
                )

            self.android_serial = device_serial

            # Auto-setup portal if enabled
            if (
                self.config.device.auto_setup
                and self.resolved_device_config.portal_mode != "disabled"
            ):
                device_obj = await adb.device(serial=device_serial)
                await ensure_portal_ready(
                    device_obj,
                    debug=self.config.logging.debug,
                    **portal_version_kwargs(ensure_portal_ready),
                )

            driver = AndroidDevice(
                serial=device_serial,
                use_tcp=self.resolved_device_config.use_tcp,
                portal_mode=self.resolved_device_config.portal_mode,
            )
            await driver.connect()

        # Framework wrappers preserve the device capability contract.
        stealth_enabled = bool(self.config.tools and self.config.tools.stealth)
        if stealth_enabled and not is_ios and not is_visual_remote:
            driver = StealthDevice(driver)

        # Existing recordings survive nested wrappers without double recording.
        if self.config.logging.save_trajectory != "none":
            if driver._recording is None:
                driver = RecordingDevice(driver)

        self.driver = driver
        self.shared_state.platform = driver.platform

        # ── 2. Create state provider ──────────────────────────────────
        if self._injected_state_provider is not None:
            self.state_provider = self._injected_state_provider
        elif self.config.agent.vision_only or is_visual_remote:
            self.state_provider = ScreenshotOnlyStateProvider(
                driver,
                vision_resize_policy=vision_resize_policy,
                use_normalized=use_normalized,
            )
        else:
            self.state_provider = driver._create_state_provider(
                StateProviderOptions(
                    vision=vision_enabled,
                    normalized=use_normalized,
                    stealth=stealth_enabled,
                    vision_resize_policy=vision_resize_policy,
                )
            )

        if self.state_provider.use_normalized:
            log_verbose(logger, "Coordinates: normalized 0-1000")

        # ── 3. Build tool registry ────────────────────────────────────
        registry, standard_tool_names = await build_tool_registry(
            device=driver,
            supported_buttons=driver.supported_buttons,
            platform="ios" if driver.platform.lower() == "ios" else "android",
            exact_app_launch=is_visual_remote,
            screenshot_only=self.state_provider.requires_coordinate_tools,
            normalized_coordinates=self.state_provider.use_normalized,
        )

        # User custom tools
        if self.user_custom_tools:
            registry.register_from_dict(self.user_custom_tools)

        # MCP tools
        if self.config.mcp and self.config.mcp.enabled:
            self.mcp_manager = MCPClientManager(self.config.mcp)
            await self.mcp_manager.discover_tools()
            mcp_tools = mcp_to_mobilerun_tools(self.mcp_manager)
            if mcp_tools:
                registry.register_from_dict(mcp_tools)

        # Capability-based filtering (deps vs driver+provider supported)
        capabilities = driver.supported_operations | self.state_provider.capabilities
        registry.disable_unsupported(capabilities)

        # Config-level filtering. ``disabled_tools=None`` means "framework
        # default"; an explicit list (even empty) is honored verbatim.
        user_disabled = self.config.tools.disabled_tools if self.config.tools else None
        explicit_disabled = user_disabled is not None
        disabled_tools = list(
            user_disabled if user_disabled is not None else DEFAULT_DISABLED_TOOLS
        )
        disabled_tools = _effective_disabled_tools(
            disabled_tools,
            self.state_provider,
            vision_enabled=action_agent_has_vision,
            explicit=explicit_disabled,
        )
        if disabled_tools:
            registry.disable(disabled_tools)

        self.registry = registry
        self.standard_tool_names = standard_tool_names

        # ── 4. Create ActionContext ────────────────────────────────────
        self.action_ctx = ActionContext(
            driver=driver,
            ui=None,  # populated each step by state_provider
            shared_state=self.shared_state,
            state_provider=self.state_provider,
            app_opener_llm=self.app_opener_llm,
            streaming=self.config.agent.streaming,
            macro_recorder=self.macro_recorder,
        )

        # ── 5. Wire up sub-agents ─────────────────────────────────────
        if self.manager_agent is not None and self.executor_agent is not None:
            self.manager_agent.action_ctx = self.action_ctx
            self.manager_agent.state_provider = self.state_provider
            self.manager_agent.registry = self.registry
            self.manager_agent.save_trajectory = self.config.logging.save_trajectory
            self.manager_agent.standard_tool_names = sorted(self.standard_tool_names)
            self.executor_agent.registry = self.registry
            self.executor_agent.action_ctx = self.action_ctx

        # ── 6. Fetch device date once ─────────────────────────────────
        try:
            self.shared_state.device_date = await driver._get_date()
        except DeviceOperationNotSupported:
            # Optional prompt context; some backends expose no device date.
            self.shared_state.device_date = ""

        # ── 7. Telemetry init event ───────────────────────────────────
        capture(
            MobileAgentInitEvent(
                goal=self.shared_state.instruction,
                llms={
                    "manager": (self.manager_llm.class_name() if self.manager_llm else "None"),
                    "executor": (self.executor_llm.class_name() if self.executor_llm else "None"),
                    "fast_agent": (
                        self.fast_agent_llm.class_name() if self.fast_agent_llm else "None"
                    ),
                    "app_opener": (
                        self.app_opener_llm.class_name() if self.app_opener_llm else "None"
                    ),
                },
                tools=",".join(sorted(standard_tool_names)),
                max_steps=self.config.agent.max_steps,
                timeout=self._init_timeout,
                vision={
                    "manager": self.config.agent.manager.vision,
                    "executor": self.config.agent.executor.vision,
                    "fast_agent": self.config.agent.fast_agent.vision,
                },
                reasoning=self.config.agent.reasoning,
                enable_tracing=self.config.tracing.enabled,
                debug=self.config.logging.debug,
                save_trajectories=self.config.logging.save_trajectory,
                runtype=self.runtype,
                custom_prompts=self._init_prompts,
            ),
            self.user_id,
            config_enabled=self.shared_state.telemetry_config_enabled,
        )

        if self.trajectory_writer is not None and self.trajectory is not None:
            self.trajectory_writer.write(self.trajectory, stage="init")

        if not self.config.agent.reasoning:
            logger.debug(
                f" Direct execution mode - executing goal: {self.shared_state.instruction}"
            )
            event = FastAgentExecuteEvent(instruction=self.shared_state.instruction)
            ctx.write_event_to_stream(event)
            return event

        logger.debug(" Reasoning mode - initializing Manager/Executor workflow")
        manager_event = ManagerInputEvent()
        ctx.write_event_to_stream(manager_event)
        return manager_event

    # ========================================================================
    # External user message injection
    # ========================================================================

    def send_user_message(self, message: str) -> QueuedUserMessage:
        queued = self.shared_state.queue_user_message(message)
        log_verbose(
            logger,
            f" External user message queued [id={queued.id}] "
            f"(queue length: {len(self.shared_state.pending_user_messages)})",
            color="cyan",
        )
        return queued

    # ========================================================================
    # execute_task — FastAgent
    # ========================================================================

    @step
    async def execute_task(self, ctx: Context, ev: FastAgentExecuteEvent) -> FastAgentResultEvent:
        """Execute a single task using FastAgent."""

        logger.debug(f" Executing task: {ev.instruction}")

        try:
            if (
                self.fast_agent_llm is None
                or self.registry is None
                or self.action_ctx is None
                or self.state_provider is None
            ):
                raise RuntimeError("FastAgent requires initialized LLM, tools and device state")
            agent = FastAgent(
                llm=self.fast_agent_llm,
                agent_config=self.config.agent,
                registry=self.registry,
                action_ctx=self.action_ctx,
                state_provider=self.state_provider,
                save_trajectory=self.config.logging.save_trajectory,
                debug=self.config.logging.debug,
                shared_state=self.shared_state,
                output_model=self.output_model,
                prompt_resolver=self.prompt_resolver,
                timeout=self.timeout,
                tracing_config=self.config.tracing,
                context_provider=self.context_provider,
            )

            handler = agent.run(
                input=ev.instruction,
            )

            async for nested_ev in handler.stream_events():
                self.handle_stream_event(nested_ev, ctx, fast_step=self.shared_state.step_number)

            result: FastAgentCompletion = await handler

            return FastAgentResultEvent(
                success=result.get("success", False),
                reason=result["reason"],
                answer=result.get("answer"),
                instruction=ev.instruction,
            )

        except DeviceDisconnectedError as e:
            logger.error(f"Device disconnected: {e}")
            return FastAgentResultEvent(
                success=False,
                reason=f"Device disconnected: {e}",
                instruction=ev.instruction,
            )

        except Exception as e:
            error = describe_error(e)
            logger.error(f"Error during task execution: {error}")
            if self.config.logging.debug:
                logger.error(traceback.format_exc())
            return FastAgentResultEvent(
                success=False, reason=f"Error: {error}", instruction=ev.instruction
            )

    @step
    async def handle_fast_agent_result(
        self, ctx: Context, ev: FastAgentResultEvent
    ) -> FinalizeEvent:
        try:
            return FinalizeEvent(success=ev.success, reason=ev.reason, answer=ev.answer)

        except Exception as e:
            error = describe_error(e)
            logger.error(f" Error during MobileAgent execution: {error}")
            if self.config.logging.debug:
                logger.error(traceback.format_exc())
            return FinalizeEvent(
                success=False,
                reason=error,
            )

    # ========================================================================
    # Manager/Executor Workflow Steps
    # ========================================================================

    @step
    async def run_manager(
        self, ctx: Context, ev: ManagerInputEvent
    ) -> ManagerPlanEvent | FinalizeEvent:
        """Run Manager planning phase."""
        if self.shared_state.step_number >= self.config.agent.max_steps:
            logger.warning(f" Reached maximum steps ({self.config.agent.max_steps})")
            pending = self.shared_state.drain_user_messages()
            if pending:
                logger.warning(f" Dropping {len(pending)} external user message(s) at max steps")
                ctx.write_event_to_stream(
                    ExternalUserMessageDroppedEvent(
                        message_ids=[m.id for m in pending],
                        reason="max_steps_reached",
                        step_number=self.shared_state.step_number,
                    )
                )
            return FinalizeEvent(
                success=False,
                reason=f"Reached maximum steps ({self.config.agent.max_steps})",
            )

        self.shared_state.step_number += 1
        log_verbose(
            logger,
            f"Step {self.shared_state.step_number}/{self.config.agent.max_steps}",
        )

        try:
            if self.manager_agent is None:
                raise RuntimeError("Manager is not configured")
            handler = self.manager_agent.run()

            async for nested_ev in handler.stream_events():
                self.handle_stream_event(nested_ev, ctx)

            result = await handler
        except DeviceDisconnectedError as e:
            logger.error(f"Device disconnected: {e}")
            return FinalizeEvent(success=False, reason=f"Device disconnected: {e}")
        except ManagerResponseValidationError as e:
            logger.error(f"Manager response validation failed: {e}")
            return FinalizeEvent(success=False, reason=str(e))

        event = ManagerPlanEvent(
            plan=result["plan"],
            current_subgoal=result["current_subgoal"],
            thought=result["thought"],
            answer=result.get("answer", ""),
            success=result.get("success"),
        )
        ctx.write_event_to_stream(event)
        return event

    @step
    async def handle_manager_plan(
        self, ctx: Context, ev: ManagerPlanEvent
    ) -> ExecutorInputEvent | FinalizeEvent | ManagerInputEvent:
        """Process Manager output and decide next step."""
        # Check for answer-type termination
        if ev.answer.strip():
            if self.shared_state.pending_user_messages:
                log_verbose(
                    logger,
                    "Manager tried to finish but external messages pending, "
                    "looping back to Manager",
                    color="cyan",
                )
                return ManagerInputEvent()
            success = ev.success if ev.success is not None else False
            self.shared_state.progress_summary = f"Answer: {ev.answer}"
            return FinalizeEvent(
                success=success,
                reason="Task completed successfully." if success else ev.answer,
                answer=ev.answer,
            )

        logger.debug(f"▶  Proceeding to Executor with subgoal: {ev.current_subgoal}")
        return ExecutorInputEvent(current_subgoal=ev.current_subgoal)

    @step
    async def run_executor(self, ctx: Context, ev: ExecutorInputEvent) -> ExecutorResultEvent:
        """Run Executor action phase."""
        logger.debug(" Running Executor for action...")

        if self.executor_agent is None:
            raise RuntimeError("Executor is not configured")
        handler = self.executor_agent.run(subgoal=ev.current_subgoal)

        async for nested_ev in handler.stream_events():
            self.handle_stream_event(nested_ev, ctx)

        result = await handler

        # Update coordination state after execution
        self.shared_state.action_history.append(result["action"])
        self.shared_state.summary_history.append(result["summary"])
        self.shared_state.action_outcomes.append(result["outcome"])
        self.shared_state.error_descriptions.append(result["error"])
        self.shared_state.last_action = result["action"]
        self.shared_state.last_summary = result["summary"]

        return ExecutorResultEvent(
            action=result["action"],
            outcome=result["outcome"],
            error=result["error"],
            summary=result["summary"],
        )

    @step
    async def handle_executor_result(
        self, ctx: Context, ev: ExecutorResultEvent
    ) -> ManagerInputEvent:
        """Process Executor result and continue."""
        err_thresh = self.shared_state.err_to_manager_thresh

        if len(self.shared_state.action_outcomes) >= err_thresh:
            latest = self.shared_state.action_outcomes[-err_thresh:]
            error_count = sum(1 for o in latest if not o)
            if error_count == err_thresh:
                logger.warning(f" Error escalation: {err_thresh} consecutive errors")
                self.shared_state.error_flag_plan = True
            else:
                if self.shared_state.error_flag_plan:
                    logger.debug(" Error resolved - resetting error flag")
                self.shared_state.error_flag_plan = False

        if self.trajectory_writer is not None and self.trajectory is not None:
            self.trajectory_writer.write(
                self.trajectory, stage=f"step_{self.shared_state.step_number}"
            )

        return ManagerInputEvent()

    # ========================================================================
    # Finalize
    # ========================================================================

    @step
    async def finalize(self, ctx: Context, ev: FinalizeEvent) -> ResultEvent:
        result = ResultEvent(
            success=ev.success,
            reason=ev.reason,
            answer=ev.answer,
            steps=self.shared_state.step_number,
            structured_output=None,
        )
        now = time.monotonic()
        workflow_deadline = await ctx.store.get(_WORKFLOW_DEADLINE_KEY, default=None)
        workflow_limit = (
            workflow_deadline - get_agent_settings().MOBILERUN_FINALIZE_SCHEDULING_RESERVE_SECONDS
            if workflow_deadline is not None
            else float("inf")
        )
        deadline = min(now + get_agent_settings().MOBILERUN_FINALIZE_BUDGET_SECONDS, workflow_limit)
        saves_trajectory = self.config.logging.save_trajectory != "none"
        has_mcp = self.mcp_manager is not None
        available = max(0.0, deadline - now)
        cleanup_reserve = 0.0
        if saves_trajectory:
            cleanup_demand = get_agent_settings().MOBILERUN_FINALIZE_TRAJECTORY_SECONDS + (
                get_agent_settings().MOBILERUN_FINALIZE_MCP_SECONDS if has_mcp else 0.0
            )
            cleanup_reserve = min(
                cleanup_demand,
                available * get_agent_settings().MOBILERUN_FINALIZE_CLEANUP_MAX_FRACTION,
            )
        elif has_mcp:
            cleanup_reserve = min(
                get_agent_settings().MOBILERUN_FINALIZE_MCP_SECONDS,
                available * get_agent_settings().MOBILERUN_FINALIZE_CLEANUP_MAX_FRACTION,
            )
        pre_cleanup_deadline = deadline - cleanup_reserve

        self.shared_state.workflow_completed = True
        ctx.write_event_to_stream(ev)

        # Extract structured output if model was provided
        if self.output_model is not None and ev.reason:
            logger.debug(" Running structured output extraction...")
            try:
                if self.structured_output_llm is None:
                    raise RuntimeError("Structured output LLM is not configured")
                structured_agent = StructuredOutputAgent(
                    llm=self.structured_output_llm,
                    pydantic_model=self.output_model,
                    answer_text=ev.answer or ev.reason,
                )
                # Avoid a nested workflow during epilog while retaining its
                # extraction-level observations and result parentage.
                completed, extraction_event = await _run_finalize_stage(
                    "structured-output",
                    _structured_output_extraction(structured_agent, ctx),
                    pre_cleanup_deadline,
                    cap=get_agent_settings().MOBILERUN_FINALIZE_BUDGET_SECONDS,
                )
                if completed and extraction_event.result["success"]:
                    result.structured_output = extraction_event.result["structured_output"]
                    logger.debug(" Structured output added to final result")
            except Exception as exc:
                logger.warning("Final structured output failed: %s", exc)

        await _run_finalize_stage(
            "telemetry-flush",
            self._flush_final_telemetry(ev),
            pre_cleanup_deadline,
            cap=get_agent_settings().MOBILERUN_FINALIZE_TELEMETRY_SECONDS,
        )

        # Capture final screenshot and UI state (independent of trajectory persistence)
        vision_any = (
            self.config.agent.manager.vision
            or self.config.agent.executor.vision
            or self.config.agent.fast_agent.vision
        )
        if vision_any or self._stream_screenshots or self.config.logging.save_trajectory != "none":
            observation_deadline = min(
                pre_cleanup_deadline,
                time.monotonic() + get_agent_settings().MOBILERUN_FINALIZE_OBSERVATION_SECONDS,
            )
            try:
                if self.action_ctx is None:
                    raise RuntimeError("Action context is not initialized")
                screenshot_operation = self.action_ctx.driver.screenshot()
                completed, screenshot = await _run_finalize_stage(
                    "screenshot",
                    screenshot_operation,
                    observation_deadline,
                    cap=get_agent_settings().MOBILERUN_FINALIZE_OBSERVATION_SECONDS,
                )
            except Exception as exc:
                logger.warning("Failed to start final screenshot: %s", exc)
                completed, screenshot = False, None
            if completed and screenshot:
                try:
                    ctx.write_event_to_stream(ScreenshotEvent(screenshot=screenshot))
                    record_langfuse_screenshot(
                        screenshot,
                        parent_span=trace.get_current_span(),
                        screenshots_enabled=self.config.tracing.langfuse_screenshots,
                        vision_enabled=vision_any,
                    )
                    logger.debug(" Final screenshot captured")
                except Exception as exc:
                    logger.warning("Failed to publish final screenshot: %s", exc)

            try:
                if self.state_provider is None:
                    raise RuntimeError("State provider is not initialized")
                completed, ui_state = await _run_finalize_stage(
                    "ui-state",
                    self.state_provider.get_final_state(),
                    observation_deadline,
                    cap=get_agent_settings().MOBILERUN_FINALIZE_OBSERVATION_SECONDS,
                )
            except Exception as exc:
                logger.warning("Failed to start final UI state: %s", exc)
                completed, ui_state = False, None
            if completed and ui_state is not None:
                try:
                    ctx.write_event_to_stream(RecordUIStateEvent(ui_state=ui_state.elements))
                    logger.debug(" Final UI state captured")
                except Exception as exc:
                    logger.warning("Failed to publish final UI state: %s", exc)

        # Save trajectory to disk
        if saves_trajectory and self.trajectory is not None and self.trajectory_writer is not None:
            trajectory_written = False
            try:
                # Prefer rich action-level macro entries; fall back to raw driver logs.
                if self.macro_recorder and self.macro_recorder.actions:
                    self.trajectory.macro = list(self.macro_recorder.actions)
                elif self.driver is not None and self.driver._recording is not None:
                    self.trajectory.macro = [
                        dict(action) for action in self.driver._recording.recorded_actions
                    ]
                self.trajectory_writer.write_final(
                    self.trajectory, self.config.logging.trajectory_gifs
                )
                trajectory_written = True
            except Exception as exc:
                logger.warning("Final trajectory persistence failed: %s", exc)
            finally:
                try:
                    mcp_reserve = (
                        min(
                            get_agent_settings().MOBILERUN_FINALIZE_MCP_SECONDS,
                            max(0.0, deadline - time.monotonic()) / 2,
                        )
                        if has_mcp
                        else 0.0
                    )
                    completed, _ = await _run_finalize_stage(
                        "trajectory-stop",
                        self.trajectory_writer.stop(),
                        deadline - mcp_reserve,
                        cap=get_agent_settings().MOBILERUN_FINALIZE_TRAJECTORY_SECONDS,
                        continue_in_background=True,
                    )
                    if trajectory_written and completed:
                        log_verbose(
                            logger,
                            f"Trajectory saved: {self.trajectory.trajectory_folder}",
                        )
                except Exception as exc:
                    logger.warning("Final trajectory shutdown failed: %s", exc)

        # Cleanup MCP connections
        if self.mcp_manager is not None:
            try:
                await _run_finalize_stage(
                    "mcp-disconnect",
                    self.mcp_manager.disconnect_all(),
                    deadline,
                    cap=get_agent_settings().MOBILERUN_FINALIZE_MCP_SECONDS,
                    continue_in_background=True,
                )
            except Exception as exc:
                logger.warning("MCP cleanup setup failed: %s", exc)

        return result

    async def _flush_final_telemetry(self, ev: FinalizeEvent) -> None:
        try:
            capture(
                MobileAgentFinalizeEvent(
                    success=ev.success,
                    reason=ev.reason,
                    steps=self.shared_state.step_number,
                    unique_packages_count=len(self.shared_state.visited_packages),
                    unique_activities_count=len(self.shared_state.visited_activities),
                ),
                self.user_id,
                config_enabled=self.shared_state.telemetry_config_enabled,
            )
        except Exception as exc:
            logger.warning("Final telemetry capture failed: %s", exc)
        await flush(config_enabled=self.shared_state.telemetry_config_enabled)

    # ========================================================================
    # Event streaming
    # ========================================================================

    def handle_stream_event(self, ev: Event, ctx: Context, *, fast_step: int | None = None) -> None:
        if not isinstance(ev, StopEvent):
            ctx.write_event_to_stream(ev)

            if self.trajectory:
                # Third-party workflow events do not implement our record contract.
                try:
                    record = ev.record
                except AttributeError:
                    self.trajectory.events.append(ev)
                else:
                    if callable(record):
                        record(self.trajectory, self.trajectory_writer, fast_step)
                    else:
                        self.trajectory.events.append(ev)

"""Opt-in local device/LLM smoke; it consumes provider tokens."""

import asyncio
import os

import pytest


@pytest.mark.smoke
@pytest.mark.smoke_llm
def test_installed_agent_operates_local_device(real_env, android_sdk_for_smoke, monkeypatch):
    serial = os.environ.get("MOBILERUN_SMOKE_SERIAL")
    provider = os.environ.get("MOBILERUN_SMOKE_LLM_PROVIDER")
    model = os.environ.get("MOBILERUN_SMOKE_LLM_MODEL")
    if not all((serial, provider, model)):
        pytest.fail(
            "Set MOBILERUN_SMOKE_SERIAL, MOBILERUN_SMOKE_LLM_PROVIDER and MOBILERUN_SMOKE_LLM_MODEL"
        )

    import sysconfig
    from pathlib import Path

    from mobilerun.core.devices import AndroidDevice

    import mobilerun_agents
    from mobilerun_agents import AgentOperator, MobileConfig
    from mobilerun_agents.config.user_config import LLMProfile

    assert (
        Path(mobilerun_agents.__file__)
        .resolve()
        .is_relative_to(Path(sysconfig.get_paths()["purelib"]).resolve())
    )

    async def run():
        device = AndroidDevice(serial=serial)
        connected = False
        try:
            await device.connect()
            connected = True
            config = MobileConfig()
            config.profiles["default"] = LLMProfile(provider=provider, model=model)
            monkeypatch.setattr("mobilerun_agents.config.loader.ConfigLoader.load", lambda: config)
            result = await device.do(
                "Open the Settings app",
                max_steps=3,
                timeout=120,
                operator=AgentOperator(),
            )
            assert result.success, result
        finally:
            try:
                if connected:
                    await device.press_button("home")
            finally:
                await device.close()

    asyncio.run(run())

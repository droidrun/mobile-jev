import shlex

from async_adbutils import adb
from mobilerun_core_local.driver.android.portal import (
    PORTAL_PACKAGE_NAME,
    portal_ime_id,
)

from mobilerun.agents.errors import KeyboardCleanupFailed


async def disable_portal_keyboard(serial: str) -> None:
    """Disable the Portal keyboard on an Android device."""
    try:
        device_obj = await adb.device(serial)
        if device_obj:
            ime = shlex.quote(portal_ime_id(PORTAL_PACKAGE_NAME))
            await device_obj.shell(f"ime disable {ime}")
    except Exception as exc:
        raise KeyboardCleanupFailed("Failed to disable Mobilerun keyboard") from exc

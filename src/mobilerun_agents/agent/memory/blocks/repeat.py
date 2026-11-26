from llama_index.core.base.llms.types import ChatMessage, ToolCallBlock

from mobilerun_agents.agent.memory.memory import SnapshotBlock
from mobilerun_agents.agent.workflows.core.snapshot import StepSnapshot


class RepeatBlock(SnapshotBlock):
    name: str = "repeat"
    priority: int = 0

    async def _aget(
        self,
        messages: list[ChatMessage] | None = None,
        snapshot: StepSnapshot | None = None,
        **kwargs: object,
    ) -> str:
        return self._guidance(snapshot, messages or []) if snapshot else ""

    async def render(self, snapshot: StepSnapshot) -> str:
        return self._guidance(snapshot, [])

    def _guidance(self, snapshot: StepSnapshot, messages: list[ChatMessage]) -> str:
        fingerprint = snapshot.fingerprint
        if not fingerprint[1]:
            return ""
        recent = tuple(reversed(snapshot._actions))
        here = [r for r in recent if r.before.fingerprint == fingerprint]
        if not here:
            return ""
        last = here[0]
        call = next(
            (
                block
                for assistant in reversed(messages)
                for block in assistant.blocks
                if isinstance(block, ToolCallBlock) and block.tool_call_id == last.call_id
            ),
            None,
        )
        name = call.tool_name if call else last.name
        args = call.tool_kwargs if call and isinstance(call.tool_kwargs, dict) else last.args
        index = args.get("index")
        label = f"{name} [{index}]" if type(index) is int else name
        target = snapshot.ui.get_element(index) if type(index) is int else None
        text = (target.get("text") or target.get("label")) if target else None
        detail = f' "{text.splitlines()[0][:80]}"' if isinstance(text, str) and text.strip() else ""
        status = "✓" if last.ok else "✗"
        change = {True: "screen changed", False: "screen unchanged", None: "screen change unknown"}
        lines = [
            f"You were on this screen at step {last.step}. Last action here: {label}{detail} {status} · {change[last.screen_changed]}."
        ]
        unchanged = repeats = 0
        for record in recent:
            if record.screen_changed is not False or record.before.fingerprint != fingerprint:
                break
            unchanged += 1
            if repeats == unchanged - 1 and (record.name, record.args) == (last.name, last.args):
                repeats += 1
        repeat_after, progress_after, cycle_after = snapshot._thresholds
        if repeats >= repeat_after:
            lines.append(f"{label} repeated {repeats}×, screen unchanged — try something else.")
        elif unchanged >= progress_after:
            lines.append(
                f"The screen was unchanged after {unchanged} consecutive actions. Try a different approach or complete."
            )
        elif (visits := _cycle_visits(snapshot)) >= cycle_after:
            lines.append(
                f"You have been on this screen {visits} times. Try a different approach or complete."
            )
        return "\n".join(lines)


def _cycle_visits(snapshot: StepSnapshot) -> int:
    current = snapshot.fingerprint
    sequence = [r.before.fingerprint for r in snapshot._actions] + [current]
    seen: set[tuple[str, str]] = set()
    previous = None
    visits = 0
    for fingerprint in sequence:
        if fingerprint == previous:
            continue
        previous = fingerprint
        if fingerprint not in seen:
            seen.add(fingerprint)
            visits = 0
        if fingerprint == current:
            visits += 1
    return visits

import logging

from llama_index.core.workflow import Event

from mobilerun.agents.logs.events import EventHandler


def test_custom_event_logs_through_its_declared_method():
    class CustomEvent(Event):
        def log(self, out: EventHandler) -> None:
            out.device_actions += 1

    handler = EventHandler()
    handler.handle(CustomEvent())
    assert handler.device_actions == 1


def test_foreign_workflow_event_uses_existing_fallback(caplog, monkeypatch):
    monkeypatch.setattr(logging.getLogger("mobilerun"), "propagate", True)
    with caplog.at_level(logging.DEBUG, logger="mobilerun"):
        EventHandler().handle(Event())
    assert caplog.records[-1].message == " Event"

import logging

from llama_index.core.workflow import Event

from mobilerun.agents.trajectory.trajectory import Trajectory


def test_print_trajectory_summary_counts_events(tmp_path, caplog):
    trajectory = Trajectory(base_path=str(tmp_path))
    trajectory.events.extend([Event(), Event()])
    logger = logging.getLogger("mobilerun")
    logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.INFO, logger="mobilerun"):
            trajectory.print_trajectory_summary({"goal": "Example", "success": True})
    finally:
        logger.removeHandler(caplog.handler)
    assert "Total steps: 2" in caplog.text
    assert "Goal: Example" in caplog.text

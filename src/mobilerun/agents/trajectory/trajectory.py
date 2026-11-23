"""
Trajectory utilities for Mobilerun agents.

This module provides the Trajectory data container and utilities for
loading and analyzing saved trajectory data.
"""

import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List

from llama_index.core.workflow import Event
from mobilerun.macro.models import (
    MACRO_SCHEMA_VERSION,
    UNSUPPORTED_SCHEMA_MESSAGE,
)

from mobilerun.agents.trajectory.writer import make_serializable

logger = logging.getLogger("mobilerun")


class Trajectory:
    def __init__(self, goal: str | None = None, base_path: str = "trajectories") -> None:
        """Initialize trajectory with incremental saving.

        Args:
            goal: The goal/prompt that this trajectory is trying to achieve
            base_path: Directory for saving (absolute or relative to cwd)
        """
        self.events: List[Event] = []
        self.screenshot_count: int = 0
        self.screenshot_queue: List[bytes] = []
        self.ui_states: List[List[Dict[str, Any]]] = []
        self.macro: List[Dict[str, Any]] = []  # populated from device._recording.actions
        self.goal = goal or "Mobilerun automation sequence"

        self.base_path = self._resolve_path(base_path)
        self.trajectory_folder = self._create_trajectory_folder()

        logger.debug(f"Trajectory folder: {self.trajectory_folder}")

    def _resolve_path(self, config_path: str) -> Path:
        """Convert config path to absolute Path."""
        path = Path(config_path)
        return path if path.is_absolute() else Path.cwd() / path

    def _create_trajectory_folder(self) -> Path:
        """Create unique trajectory folder with timestamp and UUID."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        folder = self.base_path / f"{timestamp}_{unique_id}"
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def set_goal(self, goal: str) -> None:
        """Update the goal/description for this trajectory.

        Args:
            goal: The new goal/prompt description
        """
        self.goal = goal

    def get_trajectory(self) -> List[Dict[str, Any]]:
        # Save main trajectory events
        serializable_events = []
        for event in self.events:
            event_dict = {
                "type": event.__class__.__name__,
                **{
                    k: make_serializable(v)
                    for k, v in event.__dict__.items()
                    if not k.startswith("_")
                },
            }
            serializable_events.append(event_dict)

        return serializable_events

    @staticmethod
    def load_trajectory_folder(trajectory_folder: str) -> Dict[str, Any]:
        """
        Load trajectory data from a trajectory folder.

        Args:
            trajectory_folder: Path to the trajectory folder

        Returns:
            Dictionary containing trajectory data, macro data, and file paths
        """
        result = {
            "trajectory_data": None,
            "macro_data": None,
            "gif_path": None,
            "folder_path": trajectory_folder,
        }

        try:
            # Load main trajectory
            trajectory_json_path = os.path.join(trajectory_folder, "trajectory.json")
            if os.path.exists(trajectory_json_path):
                with open(trajectory_json_path, "r") as f:
                    result["trajectory_data"] = json.load(f)
                logger.debug(f" Loaded trajectory data from {trajectory_json_path}")

            # Load macro sequence
            macro_json_path = os.path.join(trajectory_folder, "macro.json")
            if os.path.exists(macro_json_path):
                with open(macro_json_path, "r") as f:
                    result["macro_data"] = json.load(f)
                logger.debug(f" Loaded macro data from {macro_json_path}")

            # Check for GIF
            gif_path = os.path.join(trajectory_folder, "screenshots", "trajectory.gif")
            if os.path.exists(gif_path):
                result["gif_path"] = gif_path
                logger.debug(f" Found screenshot GIF at {gif_path}")

            return result

        except Exception as e:
            logger.error(f" Error loading trajectory folder {trajectory_folder}: {e}")
            return result

    @staticmethod
    def load_macro_sequence(macro_file_path: str) -> Dict[str, Any]:
        """
        Load a macro sequence from a saved macro file.

        Args:
            macro_file_path: Path to the macro JSON file (can be full path or trajectory folder)

        Returns:
            Dictionary containing the macro sequence data
        """
        # Check if it's a folder path - if so, look for macro.json inside
        if os.path.isdir(macro_file_path):
            macro_file_path = os.path.join(macro_file_path, "macro.json")

        try:
            with open(macro_file_path, "r") as f:
                macro_data = json.load(f)

            if macro_data.get("macro_schema_version") != MACRO_SCHEMA_VERSION:
                raise ValueError(UNSUPPORTED_SCHEMA_MESSAGE)

            logger.debug(
                f" Loaded macro sequence with {macro_data.get('total_actions', 0)} actions from {macro_file_path}"
            )
            return macro_data
        except FileNotFoundError:
            logger.error(f" Macro file not found: {macro_file_path}")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f" Error parsing macro file {macro_file_path}: {e}")
            return {}

    @staticmethod
    def get_macro_summary(macro_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get a summary of a macro sequence.

        Args:
            macro_data: The macro data dictionary

        Returns:
            Dictionary with statistics about the macro
        """
        if not macro_data or "actions" not in macro_data:
            return {"error": "Invalid macro data"}

        actions = macro_data["actions"]

        # Count action types
        action_types = {}
        for action in actions:
            action_type = action.get("action_type", "unknown")
            action_types[action_type] = action_types.get(action_type, 0) + 1

        # Calculate duration if timestamps are available
        timestamps = [action.get("timestamp") for action in actions if action.get("timestamp")]
        duration = max(timestamps) - min(timestamps) if len(timestamps) > 1 else 0

        return {
            "version": macro_data.get("macro_schema_version", macro_data.get("version", "unknown")),
            "description": macro_data.get("description", "No description"),
            "total_actions": len(actions),
            "action_types": action_types,
            "duration_seconds": round(duration, 2) if duration > 0 else None,
            "timestamp": macro_data.get("timestamp", "unknown"),
        }

    @staticmethod
    def print_macro_summary(macro_file_path: str) -> None:
        """
        Print a summary of a macro sequence.

        Args:
            macro_file_path: Path to the macro JSON file or trajectory folder
        """
        macro_data = Trajectory.load_macro_sequence(macro_file_path)
        if not macro_data:
            logger.warning("Could not load macro data")
            return

        summary = Trajectory.get_macro_summary(macro_data)

        logger.info("=== Macro Summary ===")
        logger.info(f"File: {macro_file_path}")
        logger.info(f"Version: {summary.get('version', 'unknown')}")
        logger.info(f"Description: {summary.get('description', 'No description')}")
        logger.info(f"Timestamp: {summary.get('timestamp', 'unknown')}")
        logger.info(f"Total actions: {summary.get('total_actions', 0)}")
        if summary.get("duration_seconds"):
            logger.info(f"Duration: {summary['duration_seconds']} seconds")
        logger.info("Action breakdown:")
        for action_type, count in summary.get("action_types", {}).items():
            logger.info(f"  - {action_type}: {count}")
        logger.info("=====================")

    @staticmethod
    def print_trajectory_folder_summary(trajectory_folder: str) -> None:
        """
        Print a comprehensive summary of a trajectory folder.

        Args:
            trajectory_folder: Path to the trajectory folder
        """
        folder_data = Trajectory.load_trajectory_folder(trajectory_folder)

        logger.info("=== Trajectory Folder Summary ===")
        logger.info(f"Folder: {trajectory_folder}")
        logger.info(
            f"Trajectory data: {'Available' if folder_data['trajectory_data'] else 'Missing'}"
        )
        logger.info(f"Macro data: {'Available' if folder_data['macro_data'] else 'Missing'}")
        logger.info(f"Screenshot GIF: {'Available' if folder_data['gif_path'] else 'Missing'}")

        if folder_data["macro_data"]:
            logger.info("\n--- Macro Summary ---")
            summary = Trajectory.get_macro_summary(folder_data["macro_data"])
            logger.info(f"Description: {summary.get('description', 'No description')}")
            logger.info(f"Total actions: {summary.get('total_actions', 0)}")
            if summary.get("duration_seconds"):
                logger.info(f"Duration: {summary['duration_seconds']} seconds")
            logger.info("Action breakdown:")
            for action_type, count in summary.get("action_types", {}).items():
                logger.info(f"  - {action_type}: {count}")

        if folder_data["trajectory_data"]:
            logger.info("\n--- Trajectory Summary ---")
            logger.info(f"Total events: {len(folder_data['trajectory_data'])}")

        logger.info("=================================")

    def print_trajectory_summary(self, trajectory_data: Dict[str, Any]) -> None:
        """
        Print a summary of a trajectory.

        Args:
            trajectory_data: The trajectory data dictionary
        """
        stats = get_trajectory_statistics(self.get_trajectory())

        logger.info("=== Trajectory Summary ===")
        logger.info(f"Goal: {trajectory_data.get('goal', 'Unknown')}")
        logger.info(f"Success: {trajectory_data.get('success', False)}")
        logger.info(f"Reason: {trajectory_data.get('reason', 'Unknown')}")
        logger.info(f"Total steps: {stats['total_steps']}")
        logger.info("Step breakdown:")
        for step_type, count in stats["step_types"].items():
            logger.info(f"  - {step_type}: {count}")
        logger.info(f"Planning steps: {stats['planning_steps']}")
        logger.info(f"Execution steps: {stats['execution_steps']}")
        logger.info(f"Successful executions: {stats['successful_executions']}")
        logger.info(f"Failed executions: {stats['failed_executions']}")
        logger.info("==========================")


def get_trajectory_statistics(
    trajectory_steps: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Get statistics about a trajectory.

    Args:
        trajectory_steps: The trajectory list of steps

    Returns:
        Dictionary with statistics about the trajectory
    """

    # Count different types of steps
    step_types = {}
    for step in trajectory_steps:
        step_type = step.get("type", "unknown")
        step_types[step_type] = step_types.get(step_type, 0) + 1

    # Count planning vs execution steps
    planning_steps = sum(
        count for step_type, count in step_types.items() if step_type.startswith("planner_")
    )
    execution_steps = sum(
        count for step_type, count in step_types.items() if step_type.startswith("fast_agent_")
    )

    # Count successful vs failed executions
    successful_executions = sum(
        1
        for step in trajectory_steps
        if step.get("type") == "fast_agent_execution" and step.get("success", False)
    )
    failed_executions = sum(
        1
        for step in trajectory_steps
        if step.get("type") == "fast_agent_execution" and not step.get("success", True)
    )

    # Return statistics
    return {
        "total_steps": len(trajectory_steps),
        "step_types": step_types,
        "planning_steps": planning_steps,
        "execution_steps": execution_steps,
        "successful_executions": successful_executions,
        "failed_executions": failed_executions,
    }


# Example usage:
"""
# Save a trajectory with a specific goal (automatically creates folder structure)
trajectory = Trajectory(goal="Open settings and check battery level")
# ... add events and screenshots to trajectory ...
folder_path = trajectory.save_trajectory()

# Or update the goal later
trajectory.set_goal("Navigate to Settings and find device info")

# Load entire trajectory folder
folder_data = Trajectory.load_trajectory_folder(folder_path)
trajectory_events = folder_data['trajectory_data']
macro_actions = folder_data['macro_data']
gif_path = folder_data['gif_path']

# Load just the macro from folder
macro_data = Trajectory.load_macro_sequence(folder_path)

# Print summaries
Trajectory.print_trajectory_folder_summary(folder_path)
Trajectory.print_macro_summary(folder_path)

# Example folder structure created:
# trajectories/
# └── trajectory_20250108_143052/
#     ├── trajectory.json      # Full trajectory events
#     ├── macro.json          # Macro sequence with goal as description
#     └── screenshots.gif     # Screenshot animation
"""

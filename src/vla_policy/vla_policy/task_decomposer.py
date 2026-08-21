#!/usr/bin/env python3
"""LLM task decomposition node for the SO-100 arm.

Uses a Large Language Model (GPT-4, LLaMA, Mistral, etc.) to
decompose a high-level natural-language instruction into a sequence
of executable subtasks for the robot.

Example:
    "Pick up the red cube and place it in the tray"
    -->
    1. Move to home position
    2. Open gripper
    3. Move above the red cube
    4. Approach the red cube
    5. Close gripper to grasp
    6. Lift the cube
    7. Move above the tray
    8. Lower to tray
    9. Open gripper to release
    10. Retreat and return home

Subscriptions
-------------
    /joint_states       sensor_msgs/JointState   Current robot state

Publications
-------------
    /task_plan          std_msgs/String          JSON subtask plan
    /task_plan/current  std_msgs/Int32           Current subtask index

Services
-------------
    /task_plan/plan     std_srvs/Trigger         Trigger new plan
    /task_plan/reset    std_srvs/Trigger         Reset to step 0

Usage
-----
    ros2 run vla_policy task_decomposer --ros-args \
        -p instruction:="pick up the red cube and place it in the tray"

    # With OpenAI:
    ros2 run vla_policy task_decomposer --ros-args \
        -p llm_provider:="openai" \
        -p instruction:="stack the blue cube on the red cube"

    # Mock mode (no LLM needed):
    ros2 run vla_policy task_decomposer --ros-args -p mock:=true
"""

import json
import threading

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_msgs.msg import String, Int32
from std_srvs.srv import Trigger


# ------------------------------------------------------------------ #
# Default task templates (used in mock mode or as fallback)            #
# ------------------------------------------------------------------ #

DEFAULT_PICK_PLACE_PLAN = [
    {
        "id": 0,
        "name": "move_home",
        "description": "Move arm to home position",
        "action": "joint_space",
        "target_joints": [0.0, 0.0, 0.0, 0.0, 0.0],
        "gripper": None,
    },
    {
        "id": 1,
        "name": "open_gripper",
        "description": "Open gripper to prepare for grasping",
        "action": "gripper",
        "target_joints": None,
        "gripper": "open",
    },
    {
        "id": 2,
        "name": "pre_grasp",
        "description": "Move above the target object",
        "action": "cartesian",
        "target_offset": [0.0, 0.0, 0.10],
        "gripper": "open",
    },
    {
        "id": 3,
        "name": "approach",
        "description": "Lower arm to approach the object",
        "action": "cartesian",
        "target_offset": [0.0, 0.0, -0.10],
        "gripper": "open",
    },
    {
        "id": 4,
        "name": "grasp",
        "description": "Close gripper to grasp the object",
        "action": "gripper",
        "target_joints": None,
        "gripper": "close",
    },
    {
        "id": 5,
        "name": "lift",
        "description": "Lift the object up",
        "action": "cartesian",
        "target_offset": [0.0, 0.0, 0.15],
        "gripper": "close",
    },
    {
        "id": 6,
        "name": "transport",
        "description": "Move above the place target",
        "action": "cartesian",
        "target_offset": [-0.30, 0.25, 0.0],
        "gripper": "close",
    },
    {
        "id": 7,
        "name": "lower",
        "description": "Lower to the place target",
        "action": "cartesian",
        "target_offset": [0.0, 0.0, -0.10],
        "gripper": "close",
    },
    {
        "id": 8,
        "name": "release",
        "description": "Open gripper to release the object",
        "action": "gripper",
        "target_joints": None,
        "gripper": "open",
    },
    {
        "id": 9,
        "name": "retreat",
        "description": "Move up and return home",
        "action": "cartesian",
        "target_offset": [0.0, 0.0, 0.15],
        "gripper": "open",
    },
]

LLM_SYSTEM_PROMPT = """You are a robot task planner for a 6-DOF robotic arm (SO-100).
Given a natural language instruction, decompose it into a sequence of subtasks.

Available actions:
- "joint_space": Move to absolute joint positions. Requires "target_joints" list of 5 floats.
- "cartesian": Move end-effector by a delta offset. Requires "target_offset" [dx, dy, dz].
- "gripper": Control the gripper. Requires "gripper" = "open" or "close".
- "wait": Wait for a condition. Requires "duration" in seconds.

Output ONLY a JSON array of subtask objects. Each object has:
- "id": integer step number (0-indexed)
- "name": short identifier
- "description": human-readable description
- "action": one of the action types above
- Any additional fields required by the action type

Example output:
[
  {"id": 0, "name": "home", "description": "Move to home", "action": "joint_space", "target_joints": [0, 0, 0, 0, 0]},
  {"id": 1, "name": "open_gripper", "description": "Open gripper", "action": "gripper", "gripper": "open"}
]
"""


class TaskDecomposerNode(Node):
    def __init__(self):
        super().__init__("task_decomposer")

        self.set_parameters([Parameter("use_sim_time", value=True)])

        # ---- Parameters -----------------------------------------------------
        self.instruction = self.declare_parameter(
            "instruction", "pick up the red cube and place it in the tray"
        ).value
        self.llm_provider = self.declare_parameter(
            "llm_provider", "mock"
        ).value  # "openai", "ollama", "mock"
        self.llm_model = self.declare_parameter(
            "llm_model", "gpt-4o"
        ).value
        self.mock = self.declare_parameter("mock", False).value

        # ---- Publishers -----------------------------------------------------
        self._plan_pub = self.create_publisher(String, "/task_plan", 10)
        self._step_pub = self.create_publisher(Int32, "/task_plan/current", 10)

        # ---- Services -------------------------------------------------------
        self.create_service(
            Trigger, "/task_plan/plan", self._plan_service
        )
        self.create_service(
            Trigger, "/task_plan/reset", self._reset_service
        )

        # ---- State ----------------------------------------------------------
        self._lock = threading.Lock()
        self._plan = []
        self._current_step = 0

    # ------------------------------------------------------------------ #
    # Services                                                            #
    # ------------------------------------------------------------------ #

    def _plan_service(self, request, response):
        self._generate_plan()
        response.success = len(self._plan) > 0
        response.message = (
            f"Generated plan with {len(self._plan)} steps"
            if self._plan else "Failed to generate plan"
        )
        return response

    def _reset_service(self, request, response):
        with self._lock:
            self._current_step = 0
        self._step_pub.publish(Int32(data=0))
        response.success = True
        response.message = "Plan reset to step 0"
        return response

    # ------------------------------------------------------------------ #
    # Plan generation                                                     #
    # ------------------------------------------------------------------ #

    def _generate_plan(self):
        if self.mock or self.llm_provider == "mock":
            self._plan = list(DEFAULT_PICK_PLACE_PLAN)
            self.get_logger().info(
                f"Generated mock plan with {len(self._plan)} steps "
                f"for: '{self.instruction}'"
            )
        else:
            plan = self._query_llm()
            if plan:
                self._plan = plan
            else:
                self.get_logger().warn("LLM failed, using default plan")
                self._plan = list(DEFAULT_PICK_PLACE_PLAN)

        self._publish_plan()

    def _query_llm(self):
        """Query an LLM for task decomposition."""
        try:
            if self.llm_provider == "openai":
                return self._query_openai()
            elif self.llm_provider == "ollama":
                return self._query_ollama()
            else:
                self.get_logger().warn(
                    f"Unknown LLM provider: {self.llm_provider}"
                )
                return None
        except Exception as exc:
            self.get_logger().error(f"LLM query failed: {exc}")
            return None

    def _query_openai(self):
        """Query OpenAI API for task decomposition."""
        try:
            import openai
        except ImportError:
            self.get_logger().fatal(
                "openai package not installed. "
                "Install with: pip install openai"
            )
            return None

        client = openai.OpenAI()
        response = client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": LLM_SYSTEM_PROMPT},
                {"role": "user", "content": self.instruction},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content
        self.get_logger().info(f"LLM response: {content[:200]}...")

        data = json.loads(content)
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "plan" in data:
            return data["plan"]
        if isinstance(data, dict) and "steps" in data:
            return data["steps"]
        return None

    def _query_ollama(self):
        """Query local Ollama for task decomposition."""
        try:
            import requests
        except ImportError:
            self.get_logger().fatal("requests package not installed")
            return None

        response = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": self.llm_model,
                "messages": [
                    {"role": "system", "content": LLM_SYSTEM_PROMPT},
                    {"role": "user", "content": self.instruction},
                ],
                "stream": False,
                "format": "json",
            },
            timeout=60,
        )

        content = response.json()["message"]["content"]
        data = json.loads(content)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ("plan", "steps", "subtasks"):
                if key in data:
                    return data[key]
        return None

    # ------------------------------------------------------------------ #
    # Publishing                                                          #
    # ------------------------------------------------------------------ #

    def _publish_plan(self):
        msg = String()
        msg.data = json.dumps(self._plan, indent=2)
        self._plan_pub.publish(msg)
        self._step_pub.publish(Int32(data=self._current_step))

        self.get_logger().info(
            f"Task plan published ({len(self._plan)} steps):"
        )
        for step in self._plan:
            self.get_logger().info(
                f"  [{step['id']}] {step['name']}: {step['description']}"
            )

    def get_current_step(self):
        """Return the current subtask or None if plan is done."""
        with self._lock:
            if self._current_step >= len(self._plan):
                return None
            return self._plan[self._current_step]

    def advance(self):
        """Move to the next subtask."""
        with self._lock:
            self._current_step += 1
            done = self._current_step >= len(self._plan)
        self._step_pub.publish(Int32(data=self._current_step))
        return done


def main():
    rclpy.init()
    node = TaskDecomposerNode()
    try:
        # Auto-generate plan on startup
        node._generate_plan()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()

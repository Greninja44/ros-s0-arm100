# SO-100 Pick & Place with a Vision-Language-Action Model

A ROS 2 (Jazzy) + Gazebo Harmonic simulation of pick-and-place on the
SO-ARM100 (so100) robot arm, driven by a Vision-Language-Action (VLA) policy
(Octo) running in the loop.

## Overview

The pipeline works like this:

```
 Gazebo Harmonic          ROS 2 (Jazzy)                 Python (Octo)
+------------------+   +-------------------------+   +------------------+
| SO-100 arm       |-->| robot_state_publisher   |-->| policy          |
| + objects        |   | /joint_states           |   |   Octo          |
| camera (sim)     |-->| image_topic             |   |   vision encoder|
|                  |   | controller_manager      |<--|   action head   |
|                  |   |   arm_controller        |   |                 |
+------------------+   +-------------------------+   +------------------+
```

1. The simulation is spawned with Gazebo Harmonic and the arm is driven by a
   `joint_trajectory_controller` (`arm_controller`).
2. A Python node runs the Octo VLA policy, which takes a natural-language
   instruction plus the current observation (camera image + joint states).
3. The policy's predicted joint positions are sent to the arm as a
   `FollowJointTrajectory` goal, closing the loop.
4. Grasping is done by closing `gripper_joint` (and optionally the second
   `gripper_finger_joint`).

## Workspace layout

```
pick_place_ws/
├── src/
│   ├── so100_description/     # URDF/xacro, meshes, Gazebo launch + controllers
│   ├── so100_moveit_config/   # MoveIt 2 config, SRDF, kinematics, RViz setup
│   └── vla_policy/            # Octo policy node + gripper + pick-and-place demo
├── build/
├── install/
└── log/
```

### so100_description

* `urdf/so100.urdf.xacro` — robot model with `gz_ros2_control` plugin
* `urdf/gazebo.xacro` — link friction / gazebo properties
* `launch/gazebo.launch.py` — spawns the arm in Gazebo, bridges the clock and
  the camera, starts RSP and loads the controllers
* `config/controllers.yaml` — `joint_state_broadcaster` + `arm_controller`
* `worlds/pick_place.world` — ground plane, sun, a red pick cube, a green place
  pad and an overhead camera (default world)
* `worlds/empty.world` — ground plane + sun

### so100_moveit_config

MoveIt 2 package generated for so100: SRDF, kinematics, joint limits,
MoveIt + RViz launch files.

### vla_policy

ROS 2 Python package with the policy and grasp nodes:

* `octo_policy_node.py` — `octo_policy` node: subscribes to the camera image and
  `/joint_states`, runs Octo on a natural-language instruction, sends the action
  chunk to `arm_controller`. Octo loads lazily; `mock:=true` tests the loop
  without a GPU/checkpoint.
* `gripper.py` — `Gripper` class (open/close via `FollowJointTrajectory`) plus
  the `gripper_demo` node.
* `pick_place.py` — `pick_place` node: full cycle (home -> pre-grasp -> approach
  -> close -> lift -> place -> open) planned through the `move_group` action.

## Prerequisites

* Ubuntu 24.04
* ROS 2 Jazzy
* Gazebo Harmonic (`ros_gz_sim`, `ros_gz_bridge`, `ros_gz_ros2_control`)
* `ros-jazzy-ros2-control` / `ros-jazzy-ros2-controllers`
* `ros-jazzy-moveit`
* xacro

## Build

```bash
cd ~/pick_place_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

## Run

### 1. Simulation only (Gazebo + controllers)

```bash
ros2 launch so100_description gazebo.launch.py
```

The default world (`pick_place.world`) spawns a red cube to pick at
`(0.25, 0.0, 0.015)` and a green place pad at `(-0.05, 0.25, 0.0025)`, matching
the `pick_place` defaults. To load the plain world instead:

```bash
ros2 launch so100_description gazebo.launch.py world:=empty.world
```

Verify the controllers are active:

```bash
ros2 control list_controllers
# should list joint_state_broadcaster and arm_controller as active
```

Check the overhead camera image (bridged from Gazebo to `/image`):

```bash
ros2 run rqt_image_view rqt_image_view
```

### 2. With MoveIt 2 (motion planning in RViz)

```bash
# terminal 1 — simulation
ros2 launch so100_description gazebo.launch.py

# terminal 2 — MoveIt + RViz
ros2 launch so100_moveit_config demo.launch.py
```

### 3. Gripper test

```bash
# terminal 1 — simulation
ros2 launch so100_description gazebo.launch.py

# terminal 2 — open/close the gripper in a loop
ros2 run vla_policy gripper_demo
```

### 4. Pick-and-place demo (MoveIt)

```bash
# terminal 1 — simulation
ros2 launch so100_description gazebo.launch.py

# terminal 2 — MoveIt move_group + RViz
ros2 launch so100_moveit_config demo.launch.py

# terminal 3 — plan + execute a full pick-and-place cycle
ros2 run vla_policy pick_place --ros-args \
  -p object_x:=0.25 -p object_y:=0.0 -p object_z:=0.05 \
  -p place_x:=-0.05 -p place_y:=0.25 -p place_z:=0.05
```

Adjust the object/place positions to match whatever you spawn in the world.

### 5. Octo VLA policy

```bash
# terminal 1 — simulation
ros2 launch so100_description gazebo.launch.py

# terminal 2 — policy node (Octo loads lazily on the first observation)
ros2 run vla_policy octo_policy --ros-args \
  -p instruction:="pick up the red cube and place it in the tray"

# without a GPU / checkpoint, validate the loop end-to-end:
ros2 run vla_policy octo_policy --ros-args -p mock:=true
```

The `mock` mode skips Octo and streams a small scripted motion so the
image -> joint_states -> arm_controller loop can be verified first.

## Key interfaces

| Topic / Action | Type | Purpose |
|---|---|---|
| `/joint_states` | `sensor_msgs/msg/JointState` | current joint positions/velocities |
| `/arm_controller/follow_joint_trajectory` | `action_msgs/msg/FollowJointTrajectory` | send trajectory goals to the arm |
| `/clock` | `rosgraph_msgs/msg/Clock` | simulation time (use_sim_time) |
| `/image` | `sensor_msgs/msg/Image` | overhead camera observation (Gazebo, bridged) |

### Joints

`shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`,
`gripper_joint`

## Roadmap

- [x] SO-100 URDF + meshes + RViz display
- [x] Gazebo simulation with ros2_control (`joint_trajectory_controller`)
- [x] MoveIt 2 config + motion planning
- [x] Gripper control (`gripper.py` / `gripper_demo`)
- [x] Octo policy inference node (`octo_policy`, image + instruction -> actions)
- [x] Pick-and-place cycle script (`pick_place`, MoveIt + gripper)
- [x] Add pick/place objects and a camera sensor to the world
- [ ] Data collection / dataset with `ros2 bag` + vision teleop
- [ ] Wire the Octo node to a real pick-and-place scene (IK action mapping)
- [ ] End-to-end pick-and-place driven by the policy in simulation
- [ ] (Optional) sim-to-real transfer to the physical SO-ARM100

## References

* [SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) — TheRobotStudio
* [Octo](https://octo-models.github.io/) — Octo VLA
* [ros_gz](https://github.com/gazebosim/ros_gz) — Gazebo ROS integration

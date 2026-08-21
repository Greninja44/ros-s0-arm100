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

The `so100_description` package lives at the workspace root (not under `src/`).
All other packages are under `src/`. Because colcon stops recursing into a
directory as soon as it finds a package there, a root-level package hides
everything under `src/` from a default `colcon build`; `colcon_defaults.yaml`
tells colcon to also crawl `src/` so all three packages are discovered.

```
pick_place_ws/                        # so100_description package root
├── colcon_defaults.yaml              # tells colcon to also crawl src/
├── CMakeLists.txt
├── package.xml
├── urdf/                             # Robot model (xacro)
│   ├── so100.urdf.xacro
│   └── gazebo.xacro
├── meshes/so100/                     # 13 STL mesh files
├── config/controllers.yaml           # ros2_control controller config
├── launch/                           # Gazebo + RViz display launch files
│   ├── gazebo.launch.py
│   └── display.launch.py
├── worlds/                           # Gazebo world files
│   ├── empty.world
│   └── pick_place.world
├── src/
│   ├── so100_moveit_config/          # MoveIt 2 config package
│   └── vla_policy/                   # Octo policy + teleop + data collection
├── build/
├── install/
└── log/
```

### so100_description

* `urdf/so100.urdf.xacro` — robot model with `gz_ros2_control` plugin, box
  collision on both gripper jaws (needed for reliable contact — see below),
  and a `DetachableJoint` plugin that lets the gripper rigidly grab the pick
  cube on demand (see `gripper.py`)
* `urdf/gazebo.xacro` — link friction / gazebo properties
* `launch/gazebo.launch.py` — spawns the arm in Gazebo, bridges the clock and
  the camera, starts RSP and loads the controllers
* `launch/display.launch.py` — lightweight RViz display (RSP + joint state
  publisher GUI, no Gazebo needed)
* `config/controllers.yaml` — `joint_state_broadcaster` + `arm_controller`
* `worlds/pick_place.world` — a table, a red pick cube at `(0.20, 0, 0.095)`,
  a green place pad at `(-0.20, 0, 0.105)` and an overhead camera publishing
  `/image`
* `worlds/empty.world` — ground plane + sun (default world)

Grasping note: the gripper's fingers are thin enough that ODE's mesh/box
collision against the pick cube doesn't reliably register contact, so
closing the jaws can't hold anything by friction alone. `gripper.py`'s
`close()`/`open()` additionally trigger a Gazebo `DetachableJoint` plugin
(rigidly attaching/detaching the cube to the gripper) as the standard
workaround for this — see the comments in `so100.urdf.xacro` and
`gripper.py` for details.

### so100_moveit_config

MoveIt 2 package generated for so100. Includes:

* `config/so100.srdf` — semantic robot description (joint groups, named poses)
* `config/kinematics.yaml` — KDL solver for the `arm` group
* `config/joint_limits.yaml` — velocity limits and scaling factors
* `config/moveit_controllers.yaml` — simple controller manager config
* `config/initial_positions.yaml` — default joint positions (all zeros)
* `config/pilz_cartesian_limits.yaml` — Pilz planner cartesian limits
* 8 launch files including `demo.launch.py` (move_group + RViz)

### vla_policy

ROS 2 Python package with the policy and grasp nodes:

* `octo_policy_node.py` — `octo_policy` node: subscribes to the camera image and
  `/joint_states`, runs Octo on a natural-language instruction, sends the action
  chunk to `arm_controller`. Supports both joint-space and Cartesian control
  modes. In Cartesian mode, the policy's end-effector deltas are converted to
  joint positions via KDL inverse kinematics. Octo loads lazily; `mock:=true`
  tests the loop without a GPU/checkpoint.
* `gripper.py` — `Gripper` class (open/close via `FollowJointTrajectory`) plus
  the `gripper_demo` node. `close()`/`open()` also attach/detach the pick
  cube via the `DetachableJoint` plugin (see the grasping note above).
* `pick_place.py` — `pick_place` node: full cycle (home -> pre-grasp -> approach
  -> close -> lift -> place -> open) planned through the `move_group` action.
* `teleop_keyboard.py` — `teleop_keyboard` node: keyboard teleoperation for the
  SO-100 arm. Controls all 5 arm joints and the gripper via key presses
  (a/d, w/s, q/e, r/f, z/x for joints, o/p for gripper).
* `collect_data.py` — `collect_data` node: records observations (camera image +
  joint states) and actions to a `rosbag2` SQLite database while teleoperating.
  Writes episode metadata in a format compatible with Octo training.
* `vla_pick_place.py` — `vla_pick_place` node: end-to-end VLA-driven
  pick-and-place. Runs the Octo policy in a closed loop with gripper control
  heuristics, IK action mapping, and episode termination.

## Prerequisites

* Ubuntu 24.04
* ROS 2 Jazzy
* Gazebo Harmonic (`ros_gz_sim`, `ros_gz_bridge`, `gz_ros2_control`)
* `ros-jazzy-ros2-control` / `ros-jazzy-ros2-controllers`
* `ros-jazzy-moveit`
* xacro

### Running under WSL2

Gazebo Transport's default multicast discovery can fail under WSL2 (a dead
`eth0` interface alongside the real one causes repeated "Network is
unreachable" / "No such device" errors and prevents Gazebo from talking to
itself). Export these before launching anything Gazebo-related:

```bash
export GZ_IP=127.0.0.1
export GZ_PARTITION=localhost
```

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

The default world is `empty.world` (ground plane + sun). To spawn the
pick-and-place scene instead:

```bash
ros2 launch so100_description gazebo.launch.py world:=pick_place.world
```

The `pick_place.world` spawns a table with a red cube at `(0.20, 0, 0.095)`
resting on top of it, and a green place pad at `(-0.20, 0, 0.105)`.

Verify the controllers are active:

```bash
ros2 control list_controllers
# should list joint_state_broadcaster and arm_controller as active
```

Check the overhead camera image (bridged from Gazebo to `/image`):

```bash
ros2 run rqt_image_view rqt_image_view
```

### 2. RViz display only (no Gazebo)

```bash
ros2 launch so100_description display.launch.py
```

Opens RViz with the robot model via `robot_state_publisher` and a joint state
publisher GUI. Useful for inspecting the URDF without spinning up Gazebo.

### 3. With MoveIt 2 (motion planning in RViz)

```bash
# terminal 1 — simulation
ros2 launch so100_description gazebo.launch.py

# terminal 2 — MoveIt + RViz
ros2 launch so100_moveit_config demo.launch.py
```

### 4. Gripper test

```bash
# terminal 1 — simulation
ros2 launch so100_description gazebo.launch.py

# terminal 2 — open/close the gripper in a loop
ros2 run vla_policy gripper_demo
```

### 5. Pick-and-place demo (MoveIt)

```bash
# terminal 1 — simulation
ros2 launch so100_description gazebo.launch.py

# terminal 2 — MoveIt move_group + RViz
ros2 launch so100_moveit_config demo.launch.py

# terminal 3 — plan + execute a full pick-and-place cycle
ros2 run vla_policy pick_place
```

`pick_place`'s `object_*`/`place_*` parameters already default to the
`pick_place.world` cube/pad positions above; pass `--ros-args -p object_x:=...`
etc. to override them if you spawn something at a different pose.

### 6. Octo VLA policy

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

### 7. Keyboard teleoperation

```bash
# terminal 1 — simulation
ros2 launch so100_description gazebo.launch.py

# terminal 2 — keyboard teleop
ros2 run vla_policy teleop_keyboard

# Keys: a/d (shoulder_pan), w/s (shoulder_lift), q/e (elbow_flex),
#        r/f (wrist_flex), z/x (wrist_roll), o/p (gripper open/close)
```

### 8. Data collection

```bash
# terminal 1 — simulation
ros2 launch so100_description gazebo.launch.py

# terminal 2 — keyboard teleop
ros2 run vla_policy teleop_keyboard

# terminal 3 — record data to rosbag
ros2 run vla_policy collect_data --ros-args \
  -p instruction:="pick up the red cube" \
  -p bag_dir:=/tmp/episode_001
```

The bag contains `/image`, `/joint_states` (observations) and `/action`
(commanded joint positions). An `episode.yaml` metadata file is written
alongside the bag.

### 9. End-to-end VLA pick-and-place

```bash
# terminal 1 — simulation
ros2 launch so100_description gazebo.launch.py

# terminal 2 — VLA pick-and-place (Octo in Cartesian mode with IK)
ros2 run vla_policy vla_pick_place --ros-args \
  -p instruction:="pick up the red cube and place it in the tray"

# mock mode (no GPU required):
ros2 run vla_policy vla_pick_place --ros-args -p mock:=true
```

The node runs the policy in a closed loop: observe -> predict -> IK ->
execute -> repeat, with automatic gripper open/close heuristics.

### 10. Octo with IK (Cartesian control)

The `octo_policy` node now supports Cartesian control mode where Octo's
end-effector deltas are converted to joint positions via KDL IK:

```bash
ros2 run vla_policy octo_policy --ros-args \
  -p control_mode:=cartesian \
  -p instruction:="pick up the red cube"
```

### 10. OpenVLA policy (drop-in Octo replacement)

```bash
# With GPU:
ros2 run vla_policy openvla_policy --ros-args \
  -p instruction:="pick up the red cube and place it in the tray"

# mock mode:
ros2 run vla_policy openvla_policy --ros-args -p mock:=true

# Cartesian mode:
ros2 run vla_policy openvla_policy --ros-args \
  -p control_mode:=cartesian \
  -p instruction:="pick up the red cube"
```

### 11. SAM 2 object segmentation

```bash
# With GPU + Grounding DINO for text-prompted detection:
ros2 run vla_policy sam2_segmentation --ros-args \
  -p prompt:="red cube"

# mock mode:
ros2 run vla_policy sam2_segmentation --ros-args -p mock:=true
```

### 12. Diffusion Policy (stochastic motion generation)

```bash
ros2 run vla_policy diffusion_policy --ros-args \
  -p instruction:="pick up the red cube"

# With a pre-trained checkpoint:
ros2 run vla_policy diffusion_policy --ros-args \
  -p checkpoint_path:=/path/to/checkpoint.pt

# mock mode:
ros2 run vla_policy diffusion_policy --ros-args -p mock:=true
```

### 13. LLM task decomposition

```bash
# Mock mode (predefined pick-and-place plan):
ros2 run vla_policy task_decomposer --ros-args \
  -p instruction:="pick up the red cube and place it in the tray"

# With OpenAI:
export OPENAI_API_KEY=your-key
ros2 run vla_policy task_decomposer --ros-args \
  -p llm_provider:=openai \
  -p instruction:="stack the blue cube on the red cube"

# With local Ollama:
ros2 run vla_policy task_decomposer --ros-args \
  -p llm_provider:=ollama \
  -p llm_model:=llama3 \
  -p instruction:="pick up the red cube"
```

### 14. Depth camera world + point cloud processing

```bash
# Terminal 1 — simulation with RGB-D camera
ros2 launch so100_description gazebo.launch.py world:=pick_place_depth.world

# Terminal 2 — point cloud processing + grasp detection
ros2 run vla_policy pointcloud_processor
```

### 15. Domain randomization world

```bash
# Terminal 1 — randomized simulation (varied lighting, objects, textures)
ros2 launch so100_description gazebo.launch.py world:=domain_randomized.world
```

### 16. Tactile sensing

```bash
# Simulated tactile feedback:
ros2 run vla_policy tactile_sensing

# With real GelSight sensor:
ros2 run vla_policy tactile_sensing --ros-args -p sensor_type:=gelsight
```

### 17. Isaac Sim (photorealistic rendering)

```bash
# Requires NVIDIA Isaac Sim installed via Omniverse
ros2 launch so100_description isaac/isaac_sim.launch.py \
  domain_randomization:=true
```

## Key interfaces

| Topic / Action | Type | Purpose |
|---|---|---|
| `/joint_states` | `sensor_msgs/msg/JointState` | current joint positions/velocities |
| `/arm_controller/follow_joint_trajectory` | `action_msgs/msg/FollowJointTrajectory` | send trajectory goals to the arm |
| `/clock` | `rosgraph_msgs/msg/Clock` | simulation time (use_sim_time) |
| `/image` | `sensor_msgs/msg/Image` | overhead camera observation (Gazebo, bridged) |
| `/depth/image` | `sensor_msgs/msg/Image` | depth camera observation (Gazebo, bridged) |
| `/pointcloud` | `sensor_msgs/msg/PointCloud2` | colored 3D point cloud from depth |
| `/sam2/mask` | `sensor_msgs/msg/Image` | SAM 2 segmentation mask |
| `/sam2/centroid` | `geometry_msgs/msg/PoseStamped` | 3D centroid of segmented object |
| `/tactile/force` | `geometry_msgs/msg/WrenchStamped` | contact force from tactile sensor |
| `/tactile/contact` | `std_msgs/msg/Bool` | contact detected |
| `/tactile/slip` | `std_msgs/msg/Float32` | slip probability |
| `/task_plan` | `std_msgs/msg/String` | JSON subtask plan from LLM |
| `/task_plan/current` | `std_msgs/msg/Int32` | current subtask index |

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
- [x] Data collection / dataset with `ros2 bag` + vision teleop
- [x] Wire the Octo node to a real pick-and-place scene (IK action mapping)
- [x] End-to-end pick-and-place driven by the policy in simulation
- [x] OpenVLA policy node (`openvla_policy`, drop-in Octo replacement)
- [x] SAM 2 zero-shot object segmentation (`sam2_segmentation`)
- [x] Diffusion Policy for stochastic action generation (`diffusion_policy`)
- [x] LLM task decomposition (`task_decomposer`, language -> subtask plan)
- [x] RGB-D depth camera support (`pick_place_depth.world`)
- [x] Domain randomization world (`domain_randomized.world`)
- [x] 3D point cloud processing + grasp pose generation (`pointcloud_processor`)
- [x] Tactile sensing integration (`tactile_sensing`, simulated/GelSight/DIGIT)
- [x] Isaac Sim configuration (`isaac_sim.yaml`, `isaac_sim.launch.py`)
- [ ] Sim-to-real transfer to the physical SO-ARM100

## References

* [SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) — TheRobotStudio
* [Octo](https://octo-models.github.io/) — Octo VLA
* [OpenVLA](https://openvla.github.io/) — OpenVLA 7B VLA
* [SAM 2](https://github.com/facebookresearch/sam2) — Segment Anything Model 2
* [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) — Chi et al.
* [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) — Text-prompted detection
* [Isaac Sim](https://developer.nvidia.com/isaac-sim) — NVIDIA photorealistic sim
* [Isaac ROS](https://developer.nvidia.com/isaac-ros) — GPU-accelerated ROS 2
* [ros_gz](https://github.com/gazebosim/ros_gz) — Gazebo ROS integration

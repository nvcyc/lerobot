# Fixes Applied to SO-ARM100 Leader-Follower Recording

## Summary

The SO-ARM100 EE recording scripts had a control instability during recording: the follower could move dramatically even though the same leader/follower setup worked in `teleoperate.py`.

The confirmed root cause was shared mutable kinematics state. The recording loop ran follower forward kinematics (FK) for dataset observations immediately before inverse kinematics (IK) for control, and both processors shared the same `RobotKinematics` instance.

---

## Confirmed Root Cause: Shared FK/IK Kinematics State

**The primary bug:** `follower_joints_to_ee` and `ee_to_follower_joints` shared `follower_kinematics_solver`.

### What Happened

`RobotKinematics.forward_kinematics()` and `RobotKinematics.inverse_kinematics()` both mutate the underlying placo robot/solver state. In `record_loop()`, the order is:

1. Read raw follower joints
2. Run `robot_observation_processor(obs)` for dataset features
3. Read leader action
4. Run leader FK to EE
5. Run follower IK to produce joint commands

When `robot_observation_processor` used follower FK with the same kinematics object as follower IK, the FK call polluted the IK solver state. That caused extreme IK targets and joint-limit saturation.

### Fix

Use a separate `RobotKinematics` instance for dataset observation FK:

```python
follower_kinematics_solver = RobotKinematics(...)
follower_observation_kinematics_solver = RobotKinematics(...)

follower_joints_to_ee = RobotProcessorPipeline(
    steps=[
        ForwardKinematicsJointsToEE(
            kinematics=follower_observation_kinematics_solver,
            motor_names=list(follower.bus.motors.keys()),
        ),
    ],
    ...
)

ee_to_follower_joints = RobotProcessorPipeline(
    steps=[
        ...,
        InverseKinematicsEEToJoints(
            kinematics=follower_kinematics_solver,
            motor_names=list(follower.bus.motors.keys()),
            initial_guess_current_joints=False,
        ),
    ],
    ...
)
```

This preserves EE observations in the dataset while keeping control IK stable.

---

## Additional Fixes Applied

### 1. ✅ **IK Initial Guess** (Continuity)
```python
# Before:
initial_guess_current_joints=True

# After:
initial_guess_current_joints=False
```
**Why:** For this setup, using the previous IK solution as the next initial guess provides smoother continuity and avoids jumping between multiple valid joint solutions.

---

### 2. ✅ **EE Bounds** (Safety Critical)
```python
# Before (Original - UNSAFE):
end_effector_bounds={"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]}

# After (Fixed):
end_effector_bounds={"min": [-0.4, -0.4, 0.0], "max": [0.4, 0.4, 0.5]}
```
**Why:** Original bounds (±1m) allowed impossible positions. Fixed to realistic SO-ARM100 workspace.

---

### 3. ✅ **Max EE Step** (Stability)
```python
# Before:
max_ee_step_m=0.10  # 10cm per step

# After:
max_ee_step_m=0.05  # 5cm per step
```
**Why:** Smaller steps improve IK convergence and stability.

---

### 4. ✅ **Hardware Configuration** (Your Setup)
```python
# Ports
follower_config.port = "/dev/ttyACM0"
leader_config.port = "/dev/ttyACM1"

# Calibration
id = "None"  # Uses None.json calibration file

# Cameras (Dual D455 RGB-only)
camera_config = {
    "left": RealSenseCameraConfig(serial_number_or_name="244422300478", ...),
    "right": RealSenseCameraConfig(serial_number_or_name="035322250292", ...)
}

# URDF path (container mount)
urdf_path="/workspace/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"

# Dataset config
HF_REPO_ID = "local/candy-picking-v1"
NUM_EPISODES = 50
TASK_DESCRIPTION = "Pick colored candies and place them in front of person"
```

---

## What We Learned

### Test Results
| Configuration | Result |
|--------------|--------|
| Direct teleop loop (`teleoperate.py`) | ✅ Stable |
| `record_teleop_debug.py --mode direct` | ✅ Stable |
| `record_teleop_debug.py --mode record-loop --observation-processor fk` | ❌ Follower moved wildly |
| `record_teleop_debug.py --mode record-loop --observation-processor identity` | ✅ Stable |
| `record_teleop_debug.py --mode record-loop --observation-processor separate-fk` | ✅ Stable |

### Conclusions
1. **Primary bug:** Shared mutable follower `RobotKinematics` between observation FK and control IK
2. **`record_loop()` itself:** Works correctly when observation FK does not share IK state
3. **Tuple format:** Works correctly
4. **Final fix:** Keep EE observation recording, but use a separate kinematics solver for follower observation FK

---

## Recommended PR to LeRobot Repo

The upstream `examples/so100_to_so100_EE/record.py` should be fixed with minimal changes:

**Required fixes:**
1. ✅ Use separate `RobotKinematics` instances for follower observation FK and follower control IK
2. ✅ Consider `initial_guess_current_joints=False` for smoother continuity on this setup
3. ✅ Use realistic EE bounds (safety)
4. ✅ Use a smaller max EE step (stability)

**NOT needed:**
- ❌ No need to remove the follower EE observation pipeline
- ❌ No need to change tuple/action format

---

## Files Modified
- ✅ `record.py` - Separate observation FK kinematics from control IK
- ✅ `record_cy.py` - Separate observation FK kinematics from control IK
- ✅ `record_candy_picking.py` - Separate observation FK kinematics from control IK and fixed missing `robot_action_to_transition` import
- ✅ `record_teleop_debug.py` - Added direct, record-loop, identity, and separate-FK validation modes

## Files for Reference
- `teleoperate_with_tuple.py` - Proves tuple format works with correct IK
- `teleoperate_original_buggy.py` - Replicates original bug for testing

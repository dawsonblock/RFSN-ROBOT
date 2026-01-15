"""
MuJoCo State Extraction Utilities
==================================
Helper functions to build ObsPacket from MuJoCo data.
"""

import mujoco as mj
import numpy as np
from rfsn.obs_packet import ObsPacket


def get_ee_pose_and_velocity(model: mj.MjModel, data: mj.MjData) -> tuple:
    """
    Get end-effector pose and velocity.
    
    Returns:
        (pos, quat, lin_vel, ang_vel)
    """
    # Get end-effector body
    ee_body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "panda_hand")
    
    # Position
    pos = data.xpos[ee_body_id].copy()
    
    # Quaternion (convert from xquat which is [w, x, y, z])
    quat = data.xquat[ee_body_id].copy()
    
    # Velocity (site or body)
    # Get linear and angular velocity from body
    lin_vel = np.zeros(3)
    ang_vel = np.zeros(3)
    
    # MuJoCo stores body velocities in data.cvel
    # cvel is [angular(3), linear(3)] for each body
    if ee_body_id < len(data.cvel):
        ang_vel = data.cvel[ee_body_id][:3].copy()
        lin_vel = data.cvel[ee_body_id][3:].copy()
    
    return pos, quat, lin_vel, ang_vel


def get_object_pose(model: mj.MjModel, data: mj.MjData, obj_name: str = "cube") -> tuple:
    """
    Get object pose.
    
    Returns:
        (pos, quat) or (None, None) if not found
    """
    try:
        obj_body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, obj_name)
        pos = data.xpos[obj_body_id].copy()
        quat = data.xquat[obj_body_id].copy()
        return pos, quat
    except:
        return None, None


def check_contacts(model: mj.MjModel, data: mj.MjData) -> dict:
    """
    Check for contacts and collisions.
    
    Returns:
        {
            'ee_contact': bool,
            'obj_contact': bool,
            'table_collision': bool,
            'self_collision': bool,
            'penetration': float
        }
    """
    result = {
        'ee_contact': False,
        'obj_contact': False,
        'table_collision': False,
        'self_collision': False,
        'penetration': 0.0
    }
    
    # Get geom IDs for checking
    try:
        cube_geom_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "cube_geom")
    except:
        cube_geom_id = -1
    
    try:
        left_finger_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "panda_finger_left_geom")
        right_finger_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "panda_finger_right_geom")
        hand_geom_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "panda_hand_geom")
    except:
        left_finger_id = right_finger_id = hand_geom_id = -1
    
    try:
        table_geom_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "table_top")
    except:
        table_geom_id = -1
    
    # Check all contacts
    max_penetration = 0.0
    panda_link_geoms = set()
    
    # Collect all panda link geom IDs
    for i in range(model.ngeom):
        try:
            name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, i)
            if name and 'panda' in name.lower():
                panda_link_geoms.add(i)
        except:
            pass
    
    for i in range(data.ncon):
        contact = data.contact[i]
        g1, g2 = contact.geom1, contact.geom2
        dist = contact.dist  # Negative means penetration
        
        # Check if this is self-collision (panda link to panda link)
        if g1 in panda_link_geoms and g2 in panda_link_geoms:
            # This is internal robot contact
            # Only count as self-collision if significant penetration
            if dist < -0.001:
                result['self_collision'] = True
            continue  # Don't count toward general penetration
        
        # Skip cube-table contact (this is normal and expected)
        if cube_geom_id >= 0 and table_geom_id >= 0:
            if (g1 == cube_geom_id and g2 == table_geom_id) or \
               (g2 == cube_geom_id and g1 == table_geom_id):
                continue  # Skip this contact, it's expected
        
        # Only count significant penetration (ignore small numerical errors)
        if dist < -0.001:  # 1mm threshold
            max_penetration = max(max_penetration, abs(dist))
        
        # EE-object contact
        if cube_geom_id >= 0:
            if (g1 == cube_geom_id and g2 in [left_finger_id, right_finger_id, hand_geom_id]) or \
               (g2 == cube_geom_id and g1 in [left_finger_id, right_finger_id, hand_geom_id]):
                result['ee_contact'] = True
                result['obj_contact'] = True
        
        # Table collision (with arm, not expected)
        # Exclude cube-table contact (this is normal and expected)
        if table_geom_id >= 0 and cube_geom_id >= 0:
            # Check if arm links collide with table (not cube or gripper)
            if (g1 == table_geom_id or g2 == table_geom_id):
                # This is a table contact
                other_geom = g2 if g1 == table_geom_id else g1
                
                # Exclude expected contacts
                if other_geom != cube_geom_id:  # Not cube-table (expected)
                    if other_geom not in [left_finger_id, right_finger_id, hand_geom_id]:  # Not gripper-table during grasp
                        result['table_collision'] = True
    
    result['penetration'] = max_penetration
    
    # Self-collision detection is already handled above (lines 116-120)
    # DO NOT override it here - safety layer depends on truthful collision signals
    
    return result


def get_gripper_state(model: mj.MjModel, data: mj.MjData) -> dict:
    """
    Get gripper state.
    
    Returns:
        {'open': bool, 'width': float}
    """
    # Get gripper joint positions
    try:
        left_q = data.qpos[7]
        right_q = data.qpos[8]
        width = abs(left_q) + abs(right_q)
        is_open = width < 0.01  # Open if fingers close to 0
        return {'open': is_open, 'width': width}
    except:
        return {'open': True, 'width': 0.0}


def compute_joint_limit_proximity(model: mj.MjModel, data: mj.MjData) -> float:
    """
    Compute proximity to joint limits (0 to 1).
    
    Returns:
        Max proximity across all joints
    """
    max_prox = 0.0
    
    for i in range(7):  # 7-DOF arm
        jnt_id = model.jnt_qposadr[i]
        q = data.qpos[jnt_id]
        q_min = model.jnt_range[i, 0]
        q_max = model.jnt_range[i, 1]
        
        # Distance from limits
        range_size = q_max - q_min
        dist_to_lower = q - q_min
        dist_to_upper = q_max - q
        
        min_dist = min(dist_to_lower, dist_to_upper)
        prox = 1.0 - (min_dist / (range_size / 2))
        prox = max(0.0, prox)
        
        max_prox = max(max_prox, prox)
    
    return max_prox


def build_obs_packet(
    model: mj.MjModel,
    data: mj.MjData,
    t: float,
    dt: float,
    mpc_converged: bool = True,
    mpc_solve_time_ms: float = 0.0,
    torque_sat_count: int = 0,
    cost_total: float = 0.0,
    task_name: str = "pick_place"
) -> ObsPacket:
    """
    Build complete ObsPacket from MuJoCo state.
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        t: Current time
        dt: Timestep
        mpc_converged: MPC convergence flag
        mpc_solve_time_ms: MPC solve time
        torque_sat_count: Number of saturated actuators
        cost_total: Total cost
        task_name: Task name
        
    Returns:
        ObsPacket
    """
    # Joint state
    q = data.qpos[:7].copy()
    qd = data.qvel[:7].copy()
    
    # End-effector
    ee_pos, ee_quat, ee_lin_vel, ee_ang_vel = get_ee_pose_and_velocity(model, data)
    
    # Gripper
    gripper = get_gripper_state(model, data)
    
    # Object
    obj_pos, obj_quat = get_object_pose(model, data, "cube")
    
    # Contacts
    contacts = check_contacts(model, data)
    
    # Joint limits
    joint_limit_prox = compute_joint_limit_proximity(model, data)
    
    return ObsPacket(
        t=t,
        dt=dt,
        q=q,
        qd=qd,
        x_ee_pos=ee_pos,
        x_ee_quat=ee_quat,
        xd_ee_lin=ee_lin_vel,
        xd_ee_ang=ee_ang_vel,
        gripper=gripper,
        x_obj_pos=obj_pos,
        x_obj_quat=obj_quat,
        x_goal_pos=None,  # Set by harness if needed
        x_goal_quat=None,
        ee_contact=contacts['ee_contact'],
        obj_contact=contacts['obj_contact'],
        table_collision=contacts['table_collision'],
        self_collision=contacts['self_collision'],
        penetration=contacts['penetration'],
        mpc_converged=mpc_converged,
        mpc_solve_time_ms=mpc_solve_time_ms,
        torque_sat_count=torque_sat_count,
        joint_limit_proximity=joint_limit_prox,
        cost_total=cost_total,
        task_name=task_name,
        success=False,
        failure_reason=None
    )

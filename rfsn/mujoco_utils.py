"""
MuJoCo State Extraction Utilities
==================================
Helper functions to build ObsPacket from MuJoCo data.
"""

import mujoco as mj
import numpy as np
from rfsn.obs_packet import ObsPacket


# Global cache for resolved IDs (initialized once per model)
_ID_CACHE = None


class GeomBodyIDs:
    """
    Cache of resolved geom/body IDs for fail-loud safety.
    
    Ensures contact checking doesn't degrade silently when XML changes.
    """
    
    def __init__(self, model: mj.MjModel):
        """
        Resolve and validate all required geom/body IDs.
        
        Raises:
            RuntimeError: If any required geom/body is missing
        """
        self.model = model
        
        # Required body IDs
        self.ee_body_id = self._resolve_body("panda_hand", "end-effector")
        self.cube_body_id = self._resolve_body("cube", "cube object")
        
        # Required geom IDs
        self.cube_geom_id = self._resolve_geom("cube_geom", "cube geometry")
        self.left_finger_id = self._resolve_geom("panda_finger_left_geom", "left finger")
        self.right_finger_id = self._resolve_geom("panda_finger_right_geom", "right finger")
        self.hand_geom_id = self._resolve_geom("panda_hand_geom", "hand palm")
        self.table_geom_id = self._resolve_geom("table_top", "table surface")
        
        # Build set of all panda link geoms (for self-collision detection)
        self.panda_link_geoms = self._collect_panda_geoms()
        
        # Validate we have all required IDs
        if not self.panda_link_geoms:
            raise RuntimeError(
                "FATAL: No panda link geoms found. "
                "Expected geoms with 'panda' in their names. "
                "Check XML model structure."
            )
        
        # Log resolved IDs once
        print("[MUJOCO_UTILS] Resolved geom/body IDs (fail-loud initialization):")
        print(f"  EE body:         {self.ee_body_id} (panda_hand)")
        print(f"  Cube body:       {self.cube_body_id} (cube)")
        print(f"  Cube geom:       {self.cube_geom_id} (cube_geom)")
        print(f"  Left finger:     {self.left_finger_id} (panda_finger_left_geom)")
        print(f"  Right finger:    {self.right_finger_id} (panda_finger_right_geom)")
        print(f"  Hand geom:       {self.hand_geom_id} (panda_hand_geom)")
        print(f"  Table geom:      {self.table_geom_id} (table_top)")
        print(f"  Panda link geoms: {len(self.panda_link_geoms)} geoms")
        
    def _resolve_body(self, name: str, description: str) -> int:
        """Resolve body ID with clear error message."""
        try:
            body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, name)
            if body_id < 0:
                raise ValueError(f"Invalid body ID: {body_id}")
            return body_id
        except Exception as e:
            raise RuntimeError(
                f"FATAL: Required body '{name}' ({description}) not found in model. "
                f"Error: {e}. Check XML model structure."
            )
    
    def _resolve_geom(self, name: str, description: str) -> int:
        """Resolve geom ID with clear error message."""
        try:
            geom_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, name)
            if geom_id < 0:
                raise ValueError(f"Invalid geom ID: {geom_id}")
            return geom_id
        except Exception as e:
            raise RuntimeError(
                f"FATAL: Required geom '{name}' ({description}) not found in model. "
                f"Error: {e}. Check XML model structure."
            )
    
    def _collect_panda_geoms(self) -> set:
        """Collect all panda link geom IDs."""
        panda_geoms = set()
        for i in range(self.model.ngeom):
            try:
                name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_GEOM, i)
                if name and 'panda' in name.lower():
                    panda_geoms.add(i)
            except:
                pass  # Skip invalid geoms
        return panda_geoms


def init_id_cache(model: mj.MjModel):
    """
    Initialize the global ID cache with resolved geom/body IDs.
    
    Must be called once before using other functions.
    Raises RuntimeError if any required geom/body is missing.
    
    Args:
        model: MuJoCo model
    """
    global _ID_CACHE
    _ID_CACHE = GeomBodyIDs(model)


def get_id_cache() -> GeomBodyIDs:
    """
    Get the initialized ID cache.
    
    Returns:
        GeomBodyIDs instance
        
    Raises:
        RuntimeError: If cache not initialized
    """
    global _ID_CACHE
    if _ID_CACHE is None:
        raise RuntimeError(
            "FATAL: ID cache not initialized. "
            "Call init_id_cache(model) before using mujoco_utils functions."
        )
    return _ID_CACHE


def get_ee_pose_and_velocity(model: mj.MjModel, data: mj.MjData) -> tuple:
    """
    Get end-effector pose and velocity.
    
    Returns:
        (pos, quat, lin_vel, ang_vel)
    """
    # Get end-effector body (use cached ID)
    ids = get_id_cache()
    ee_body_id = ids.ee_body_id
    
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
    Check for contacts and collisions using cached geom IDs.
    
    Uses pre-resolved IDs for fail-loud correctness.
    
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
    
    # Use cached IDs (fail-loud if not initialized)
    ids = get_id_cache()
    
    # Check all contacts
    max_penetration = 0.0
    
    for i in range(data.ncon):
        contact = data.contact[i]
        g1, g2 = contact.geom1, contact.geom2
        dist = contact.dist  # Negative means penetration
        
        # Check if this is self-collision (panda link to panda link)
        if g1 in ids.panda_link_geoms and g2 in ids.panda_link_geoms:
            # This is internal robot contact
            # Only count as self-collision if significant penetration
            if dist < -0.001:
                result['self_collision'] = True
            continue  # Don't count toward general penetration
        
        # Skip cube-table contact (this is normal and expected)
        if (g1 == ids.cube_geom_id and g2 == ids.table_geom_id) or \
           (g2 == ids.cube_geom_id and g1 == ids.table_geom_id):
            continue  # Skip this contact, it's expected
        
        # Only count significant penetration (ignore small numerical errors)
        if dist < -0.001:  # 1mm threshold
            max_penetration = max(max_penetration, abs(dist))
        
        # EE-object contact
        finger_geoms = {ids.left_finger_id, ids.right_finger_id, ids.hand_geom_id}
        if (g1 == ids.cube_geom_id and g2 in finger_geoms) or \
           (g2 == ids.cube_geom_id and g1 in finger_geoms):
            result['ee_contact'] = True
            result['obj_contact'] = True
        
        # Table collision (with arm, not expected)
        if g1 == ids.table_geom_id or g2 == ids.table_geom_id:
            # This is a table contact
            other_geom = g2 if g1 == ids.table_geom_id else g1
            
            # Exclude expected contacts
            if other_geom != ids.cube_geom_id:  # Not cube-table (expected)
                if other_geom not in finger_geoms:  # Not gripper-table during grasp
                    result['table_collision'] = True
    
    result['penetration'] = max_penetration
    
    return result


def self_test_contact_parsing(model: mj.MjModel, data: mj.MjData):
    """
    Self-test to validate contact parsing works correctly.
    
    Runs one forward step and checks that contacts dict has all expected keys.
    
    Raises:
        RuntimeError: If contact parsing fails
    """
    try:
        # Run one forward step
        mj.mj_forward(model, data)
        
        # Check contact parsing
        contacts = check_contacts(model, data)
        
        # Validate all keys present
        required_keys = ['ee_contact', 'obj_contact', 'table_collision', 
                        'self_collision', 'penetration']
        for key in required_keys:
            if key not in contacts:
                raise RuntimeError(
                    f"FATAL: Contact parsing missing key '{key}'. "
                    "Contact dict is incomplete."
                )
        
        # Validate types
        for key in ['ee_contact', 'obj_contact', 'table_collision', 'self_collision']:
            if not isinstance(contacts[key], bool):
                raise RuntimeError(
                    f"FATAL: Contact key '{key}' has wrong type {type(contacts[key])}, "
                    "expected bool."
                )
        
        if not isinstance(contacts['penetration'], (int, float)):
            raise RuntimeError(
                f"FATAL: Contact key 'penetration' has wrong type {type(contacts['penetration'])}, "
                "expected float."
            )
        
        print("[MUJOCO_UTILS] Self-test PASSED: Contact parsing validated")
        
    except Exception as e:
        raise RuntimeError(
            f"FATAL: Contact parsing self-test failed: {e}. "
            "Cannot safely proceed with contact-based safety."
        )


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

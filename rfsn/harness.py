"""
RFSN Unified Harness
====================
Main integration point: wraps low-level controller with RFSN executive layer.

IMPORTANT: Despite naming conventions, this currently uses PD control + MuJoCo inverse
dynamics (mj_inverse), NOT true Model Predictive Control (MPC). The "MPC knobs" from
RFSN profiles actually control PD gains (KP/KD) and torque scaling.

The FastMPCController exists but is not integrated into this control loop.
To use actual MPC, FastMPCController.compute() would need to be called to generate
trajectory references.

Supports 3 modes:
1. "mpc_only" (baseline): PD + inverse dynamics with fixed joint target
2. "rfsn": RFSN state machine generates EE targets, converted via IK to joint targets
3. "rfsn_learning": Same as rfsn, with UCB-based profile learning
"""

import mujoco as mj
import numpy as np
import time
from typing import Optional

from rfsn.obs_packet import ObsPacket
from rfsn.decision import RFSNDecision
from rfsn.state_machine import RFSNStateMachine
from rfsn.profiles import ProfileLibrary
from rfsn.learner import SafeLearner
from rfsn.safety import SafetyClamp
from rfsn.logger import RFSNLogger
from rfsn.mujoco_utils import build_obs_packet


class RFSNHarness:
    """
    Unified harness for PD control + RFSN integration.
    
    Control Law: PD control in joint space + MuJoCo inverse dynamics (mj_inverse)
    
    Modes:
    - "mpc_only": Baseline PD control with fixed joint targets
    - "rfsn": RFSN state machine generates EE targets → IK → joint targets → PD
    - "rfsn_learning": Same as "rfsn" with UCB profile learning
    
    CRITICAL: Profile "MPC Parameters" Are Actually PD Control Proxies
    ===================================================================
    Despite naming, profiles do NOT configure true MPC. They map to PD control:
    
    - horizon_steps: PROXY for IK iteration count (more iterations = finer convergence)
                     NOT an MPC prediction horizon
    
    - Q_diag[0:7]:   PROXY for PD position gains (KP_scale = sqrt(Q_pos / 50.0))
                     Higher Q → higher KP → stiffer position tracking
    
    - Q_diag[7:14]:  PROXY for PD velocity gains (KD_scale = sqrt(Q_vel / 10.0))
                     Higher Q_vel → higher KD → more damping
    
    - R_diag:        NOT USED in current implementation
                     Reserved for future control effort penalty
    
    - du_penalty:    NOT USED in current implementation
                     Reserved for future rate limiting or smoothing
    
    - max_tau_scale: Direct torque multiplier (≤1.0 for safety)
                     Reduces available torque to prevent aggressive motion
    
    Learning selects among these proxy profiles, NOT raw control actions.
    """
    
    def __init__(
        self,
        model: mj.MjModel,
        data: mj.MjData,
        mode: str = "mpc_only",
        task_name: str = "pick_place",
        logger: Optional[RFSNLogger] = None
    ):
        """
        Initialize RFSN harness.
        
        Args:
            model: MuJoCo model
            data: MuJoCo data
            mode: Control mode ("mpc_only", "rfsn", "rfsn_learning")
            task_name: Task name
            logger: Optional logger
        """
        assert mode in ["mpc_only", "rfsn", "rfsn_learning"]
        
        self.model = model
        self.data = data
        self.mode = mode
        self.task_name = task_name
        self.logger = logger
        
        self.dt = model.opt.timestep
        self.t = 0.0
        self.step_count = 0
        
        # PD gains for baseline MPC
        self.KP = np.array([300.0, 300.0, 300.0, 300.0, 150.0, 100.0, 50.0])
        self.KD = np.array([60.0, 60.0, 60.0, 60.0, 30.0, 20.0, 10.0])
        
        # RFSN components (initialized only if needed)
        self.rfsn_enabled = mode in ["rfsn", "rfsn_learning"]
        
        if self.rfsn_enabled:
            self.profile_library = ProfileLibrary()
            self.state_machine = RFSNStateMachine(task_name, self.profile_library)
            self.safety_clamp = SafetyClamp()
            
            if mode == "rfsn_learning":
                self.learner = SafeLearner(self.profile_library)
            else:
                self.learner = None
        else:
            self.profile_library = None
            self.state_machine = None
            self.safety_clamp = None
            self.learner = None
        
        # Baseline target
        self.baseline_target_q = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
        
        # Episode tracking
        self.episode_active = False
        self.obs_history = []
        self.decision_history = []
        self.initial_cube_z = None  # Track initial cube height for grasp quality
        
    def start_episode(self):
        """Start a new episode."""
        self.t = 0.0
        self.step_count = 0
        self.episode_active = True
        self.obs_history = []
        self.decision_history = []
        self.initial_cube_z = None
        
        if self.rfsn_enabled:
            self.state_machine.reset()
    
    def step(self) -> ObsPacket:
        """
        Execute one control step.
        
        Returns:
            ObsPacket with current observation
        """
        # Build observation
        t_mpc_start = time.perf_counter()
        
        obs = build_obs_packet(
            self.model,
            self.data,
            t=self.t,
            dt=self.dt,
            task_name=self.task_name
        )
        
        # Track initial cube height on first step
        if self.initial_cube_z is None and obs.x_obj_pos is not None:
            self.initial_cube_z = obs.x_obj_pos[2]
        
        # Generate decision
        if self.rfsn_enabled:
            # RFSN mode: state machine generates decision
            profile_variant = None
            if self.learner:
                profile_variant = self.learner.select_profile(
                    self.state_machine.current_state,
                    self.t,
                    safety_poison_check=self.safety_clamp.is_poisoned
                )
            
            # Compute grasp quality for GRASP state
            grasp_quality = None
            if self.state_machine.current_state == "GRASP":
                grasp_quality = self._check_grasp_quality(obs, self.initial_cube_z)
            
            decision = self.state_machine.step(obs, profile_override=profile_variant,
                                              grasp_quality=grasp_quality)
            
            # Apply safety clamps
            decision = self.safety_clamp.apply(decision, obs)
            
            # Convert EE target to joint target (IK stub)
            q_target = self._ee_target_to_joint_target(decision)
        else:
            # Baseline MPC mode: fixed joint target
            decision = None
            q_target = self.baseline_target_q.copy()
        
        # Compute control (inverse dynamics)
        tau = self._inverse_dynamics_control(obs.q, obs.qd, q_target, decision)
        
        # Apply control
        self.data.ctrl[:7] = tau
        
        # Gripper control with proper grasp detection
        if self.rfsn_enabled and decision:
            if decision.task_mode in ["GRASP", "LIFT", "TRANSPORT", "PLACE"]:
                # Close gripper with force control
                self.data.ctrl[7] = -80.0  # Close left finger
                self.data.ctrl[8] = 80.0   # Close right finger
            elif decision.task_mode in ["REACH_PREGRASP", "REACH_GRASP"]:
                # Pre-open gripper for approach
                self.data.ctrl[7] = 40.0   # Open left finger
                self.data.ctrl[8] = -40.0  # Open right finger
            else:
                # Neutral/open position
                self.data.ctrl[7] = 20.0
                self.data.ctrl[8] = -20.0
        else:
            # MPC-only mode: keep gripper open
            self.data.ctrl[7] = 20.0
            self.data.ctrl[8] = -20.0
        
        # Step simulation
        mj.mj_step(self.model, self.data)
        
        # Update MPC diagnostics
        mpc_solve_time = (time.perf_counter() - t_mpc_start) * 1000  # ms
        obs.mpc_solve_time_ms = mpc_solve_time
        
        # Count torque saturation
        torque_sat_count = np.sum(np.abs(tau) >= 86.5)  # Near 87 limit
        obs.torque_sat_count = torque_sat_count
        
        # Log
        if self.logger:
            if decision:
                self.logger.log_step(obs, decision)
            elif self.episode_active:
                # In MPC-only mode, create a dummy decision for logging
                dummy_decision = RFSNDecision(
                    task_mode="IDLE",  # Use IDLE for baseline
                    x_target_pos=obs.x_ee_pos.copy(),
                    x_target_quat=obs.x_ee_quat.copy(),
                    horizon_steps=10,
                    Q_diag=np.ones(14) * 50.0,
                    R_diag=0.01 * np.ones(7),
                    terminal_Q_diag=np.ones(14) * 500.0,
                    du_penalty=0.01,
                    max_tau_scale=1.0,
                    contact_policy="AVOID",
                    confidence=1.0,
                    reason="baseline_mpc",
                    rollback_token="mpc_baseline"
                )
                self.logger.log_step(obs, dummy_decision)
        
        if self.episode_active:
            self.obs_history.append(obs)
            if decision:
                self.decision_history.append(decision)
        
        self.t += self.dt
        self.step_count += 1
        
        return obs
    
    def end_episode(self, success: bool = False, failure_reason: str = None):
        """End current episode and update learning with safety coupling."""
        self.episode_active = False
        
        # Update learner if enabled
        if self.learner and len(self.obs_history) > 0:
            score, violations = self.learner.compute_score(
                self.obs_history, 
                self.decision_history
            )
            
            # Track which (state, profile) pairs were used and count severe events per pair
            state_profile_usage = {}  # (state, profile) -> {'count': int, 'severe_events': int}
            
            for i, decision in enumerate(self.decision_history):
                # Extract profile name from rollback token
                profile_name = decision.rollback_token.split('_')[1] if '_' in decision.rollback_token else 'base'
                key = (decision.task_mode, profile_name)
                
                if key not in state_profile_usage:
                    state_profile_usage[key] = {'count': 0, 'severe_events': 0}
                
                state_profile_usage[key]['count'] += 1
                
                # Count severe events at this step
                if i < len(self.obs_history):
                    obs = self.obs_history[i]
                    if (obs.self_collision or obs.table_collision or 
                        obs.penetration > 0.05 or obs.torque_sat_count >= 5):
                        state_profile_usage[key]['severe_events'] += 1
            
            # Update stats and poison profiles with repeated severe events
            for (state, profile), usage_info in state_profile_usage.items():
                # Update learner statistics
                profile_score = score / len(state_profile_usage)  # Distribute score
                profile_violations = usage_info['severe_events']
                
                self.learner.update_stats(state, profile, profile_score, profile_violations, self.t)
                
                # Check if this profile should be poisoned (2+ severe events in last 5 uses)
                stats = self.learner.stats.get((state, profile))
                if stats and stats.N >= 5 and hasattr(stats, 'recent_scores') and len(stats.recent_scores) >= 5:
                    recent_severe_count = sum(1 for s in stats.recent_scores[-5:] if s < -5.0)
                    if recent_severe_count >= 2:
                        # Poison this profile to prevent future selection
                        self.safety_clamp.poison_profile(state, profile)
                        print(f"[HARNESS] Poisoned ({state}, {profile}) due to repeated severe events")

        
        if self.logger:
            self.logger.end_episode(success, failure_reason)

    
    def _ee_target_to_joint_target(self, decision: RFSNDecision, use_orientation: bool = None) -> np.ndarray:
        """
        Convert end-effector target to joint target using damped least squares IK.
        
        Args:
            decision: Decision containing target pose and horizon_steps
            use_orientation: If True, include orientation in IK. If None, auto-decide based on state.
        
        Uses MuJoCo Jacobian and iterative pose-based (position + orientation) IK with damping.
        Orientation is soft-weighted and optional per state for stability.
        
        PROXY MAPPING: decision.horizon_steps → IK max_iterations
        Higher horizon → more IK iterations → finer convergence (but slower)
        """
        q_current = self.data.qpos[:7].copy()
        
        # Get end-effector body ID
        ee_body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "panda_hand")
        
        # Decide whether to use orientation based on state (if not explicitly specified)
        if use_orientation is None:
            # Enable orientation for states where precise pose matters
            use_orientation = decision.task_mode in ["GRASP", "PLACE", "REACH_GRASP"]
        
        # Iterative IK with damped least squares
        q_ik = q_current.copy()
        alpha = 0.5  # Step size
        damping_pos = 0.01  # Position damping for stability
        damping_rot = 0.05  # Higher rotation damping (orientation is soft)
        
        # PROXY: Use horizon_steps as max IK iterations (clamped for safety)
        # More iterations = more precise convergence = "longer planning horizon" metaphor
        max_iterations = min(max(decision.horizon_steps, 5), 20)  # Clamp to [5, 20]
        
        pos_tolerance = 0.01  # 1cm
        ori_tolerance = 0.1   # Quaternion distance tolerance
        
        # Orientation weight (soft constraint)
        ori_weight = 0.3  # Lower weight than position (soft orientation)
        
        for iteration in range(max_iterations):
            # Update MuJoCo data with current joint positions
            data_temp = mj.MjData(self.model)
            data_temp.qpos[:] = self.data.qpos
            data_temp.qpos[:7] = q_ik
            mj.mj_forward(self.model, data_temp)
            
            # Get current EE pose
            ee_pos_current = data_temp.xpos[ee_body_id].copy()
            ee_quat_current = data_temp.xquat[ee_body_id].copy()  # [w, x, y, z]
            
            # Compute position error
            pos_error = decision.x_target_pos - ee_pos_current
            
            # Compute orientation error (axis-angle from quaternion difference)
            ori_error = np.zeros(3)
            if use_orientation:
                ori_error = self._quaternion_error(ee_quat_current, decision.x_target_quat)
            
            # Check convergence
            pos_converged = np.linalg.norm(pos_error) < pos_tolerance
            ori_converged = np.linalg.norm(ori_error) < ori_tolerance if use_orientation else True
            if pos_converged and ori_converged:
                break
            
            # Compute Jacobians
            jacp = np.zeros((3, self.model.nv))  # Position Jacobian
            jacr = np.zeros((3, self.model.nv))  # Rotation Jacobian
            mj.mj_jacBodyCom(self.model, data_temp, jacp, jacr, ee_body_id)
            
            # Extract arm joints only (first 7 DOF)
            J_pos = jacp[:, :7]
            J_rot = jacr[:, :7]
            
            if use_orientation:
                # Combine position and orientation Jacobians
                # Stack: [position (3), orientation (3)]
                J = np.vstack([J_pos, ori_weight * J_rot])
                error = np.concatenate([pos_error, ori_weight * ori_error])
                
                # Damped least squares with combined Jacobian
                JJT = J @ J.T
                damping_matrix = np.diag([damping_pos**2] * 3 + [damping_rot**2] * 3)
                dq = J.T @ np.linalg.solve(JJT + damping_matrix, error)
            else:
                # Position-only IK (original behavior)
                JJT = J_pos @ J_pos.T
                damping_matrix = damping_pos**2 * np.eye(3)
                dq = J_pos.T @ np.linalg.solve(JJT + damping_matrix, pos_error)
            
            # Update joint angles with step size
            q_ik += alpha * dq
            
            # Clamp to joint limits
            for i in range(7):
                q_min = self.model.jnt_range[i, 0]
                q_max = self.model.jnt_range[i, 1]
                q_ik[i] = np.clip(q_ik[i], q_min, q_max)
        
        return q_ik
    
    def _quaternion_error(self, q_current: np.ndarray, q_target: np.ndarray) -> np.ndarray:
        """
        Compute orientation error as axis-angle from quaternion difference.
        
        Args:
            q_current: Current quaternion [w, x, y, z]
            q_target: Target quaternion [w, x, y, z]
        
        Returns:
            axis-angle error (3,) for use in Jacobian IK
        """
        # Ensure quaternions are normalized
        q_current = q_current / np.linalg.norm(q_current)
        q_target = q_target / np.linalg.norm(q_target)
        
        # Compute quaternion difference: q_error = q_target * q_current^{-1}
        # Quaternion conjugate (inverse for unit quaternions)
        q_current_conj = np.array([q_current[0], -q_current[1], -q_current[2], -q_current[3]])
        
        # Quaternion multiplication: q_error = q_target * q_current_conj
        w1, x1, y1, z1 = q_target
        w2, x2, y2, z2 = q_current_conj
        
        q_error = np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,  # w
            w1*x2 + x1*w2 + y1*z2 - z1*y2,  # x
            w1*y2 - x1*z2 + y1*w2 + z1*x2,  # y
            w1*z2 + x1*y2 - y1*x2 + z1*w2   # z
        ])
        
        # Convert to axis-angle (small angle approximation for stability)
        # For small rotations: axis * angle ≈ 2 * [x, y, z] components
        # This is valid when w ≈ 1 (small rotation)
        axis_angle = 2.0 * q_error[1:4]
        
        # Clamp to prevent large corrections
        max_angle = 0.5  # ~28 degrees max per iteration
        angle_norm = np.linalg.norm(axis_angle)
        if angle_norm > max_angle:
            axis_angle = axis_angle * (max_angle / angle_norm)
        
        return axis_angle
    
    def _inverse_dynamics_control(
        self,
        q: np.ndarray,
        qd: np.ndarray,
        q_target: np.ndarray,
        decision: Optional[RFSNDecision]
    ) -> np.ndarray:
        """
        Compute control torques using inverse dynamics.
        
        Uses MuJoCo's mj_inverse to compute required torques for PD control.
        
        Profile Parameter Mapping (EXPLICIT):
        =====================================
        - Q_diag[0:7]  → KP_scale = sqrt(Q_pos / 50.0) → Position stiffness
        - Q_diag[7:14] → KD_scale = sqrt(Q_vel / 10.0) → Velocity damping
        - R_diag       → NOT USED (future: control effort penalty)
        - du_penalty   → NOT USED (future: acceleration smoothing)
        - max_tau_scale→ DIRECT multiplier on output torques (safety limiter)
        
        This is NOT MPC despite profile naming. It's gain-scheduled PD control.
        """
        # Map profile "Q" parameters to PD gains (EXPLICIT PROXY MAPPING)
        if decision:
            # Q_diag[0:7] controls position stiffness via KP scaling
            kp_scale = np.sqrt(decision.Q_diag[:7] / 50.0)  # Normalized to base=50
            KP_local = self.KP * kp_scale
            
            # Q_diag[7:14] controls velocity damping via KD scaling
            kd_scale = np.sqrt(decision.Q_diag[7:14] / 10.0)  # Normalized to base=10
            KD_local = self.KD * kd_scale
            
            # Note: R_diag and du_penalty are NOT used in current implementation
            # They exist for potential future extensions (control effort, smoothing)
        else:
            KP_local = self.KP
            KD_local = self.KD
        
        # PD control law
        q_error = q_target - q
        dq_desired = KP_local * q_error
        
        # Create temp data for inverse dynamics
        data_temp = mj.MjData(self.model)
        data_temp.qpos[:] = self.data.qpos
        data_temp.qvel[:] = self.data.qvel
        
        # Set desired acceleration (PD output)
        qacc_full = np.zeros(self.model.nv)
        qacc_full[:7] = dq_desired - KD_local * qd
        data_temp.qacc[:] = qacc_full
        
        # Compute inverse dynamics (maps acceleration to torques)
        mj.mj_inverse(self.model, data_temp)
        tau = data_temp.qfrc_inverse[:7].copy()
        
        # Apply torque scale (PROXY: max_tau_scale as safety limiter)
        if decision:
            tau *= decision.max_tau_scale
        
        # Saturate
        tau = np.clip(tau, -87.0, 87.0)
        
        return tau
    
    def _check_grasp_quality(self, obs: ObsPacket, initial_cube_z: float = None) -> dict:
        """
        Check grasp quality based on contacts, gripper state, and cube attachment.
        
        Args:
            obs: Current observation
            initial_cube_z: Initial cube height (for attachment detection)
        
        Returns:
            {
                'has_contact': bool - whether fingers are in contact with object
                'is_stable': bool - whether grasp is stable (both fingers, low motion)
                'is_attached': bool - whether cube is following EE (attachment proxy)
                'quality': float - grasp quality score 0-1
            }
        """
        # Grasp quality thresholds
        GRIPPER_CLOSED_WIDTH = 0.06  # Gripper width threshold for "closed" (meters)
        LOW_VELOCITY_THRESHOLD = 0.1  # EE velocity threshold for "stable" (m/s)
        LIFT_HEIGHT_THRESHOLD = 0.02  # Minimum lift to confirm attachment (meters)
        
        result = {
            'has_contact': obs.obj_contact and obs.ee_contact,
            'is_stable': False,
            'is_attached': False,
            'quality': 0.0
        }
        
        # Check if both fingers are in contact
        if not result['has_contact']:
            return result
        
        # Check gripper width (closed enough)
        gripper_width = obs.gripper.get('width', 0.0)
        is_closed = gripper_width < GRIPPER_CLOSED_WIDTH
        
        # Check relative motion (EE velocity as proxy for grasp stability)
        # Note: ObsPacket doesn't include object velocity, so we use EE velocity
        # which should be low during stable grasp
        if obs.x_obj_pos is not None:
            ee_vel_norm = np.linalg.norm(obs.xd_ee_lin)
            is_low_motion = ee_vel_norm < LOW_VELOCITY_THRESHOLD
            
            # Check cube attachment: cube should have lifted from initial position
            if initial_cube_z is not None:
                cube_lifted = obs.x_obj_pos[2] > (initial_cube_z + LIFT_HEIGHT_THRESHOLD)
                result['is_attached'] = cube_lifted
        else:
            is_low_motion = True
        
        # Grasp is stable if closed, low motion, and has contact
        result['is_stable'] = is_closed and is_low_motion and result['has_contact']
        
        # Compute quality score
        quality = 0.0
        if result['has_contact']:
            quality += 0.3  # Contact
        if is_closed:
            quality += 0.25  # Gripper closed
        if is_low_motion:
            quality += 0.2  # Low velocity
        if result['is_attached']:
            quality += 0.25  # Cube lifted (strong indicator)
        
        result['quality'] = quality
        
        return result
    
    def get_stats(self) -> dict:
        """Get harness statistics."""
        stats = {
            'mode': self.mode,
            'step_count': self.step_count,
            'time': self.t,
        }
        
        if self.safety_clamp:
            stats['safety'] = self.safety_clamp.get_stats()
        
        if self.learner:
            stats['learning'] = self.learner.get_stats_summary()
        
        return stats

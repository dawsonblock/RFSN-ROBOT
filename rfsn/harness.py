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
    
    RFSN Profile Knobs Actually Control:
    - horizon_steps: Not used in current PD implementation
    - Q_diag: Scales PD position gains (KP)
    - R_diag: Not directly used (could scale KD in future)
    - max_tau_scale: Multiplies output torques
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
        
    def start_episode(self):
        """Start a new episode."""
        self.t = 0.0
        self.step_count = 0
        self.episode_active = True
        self.obs_history = []
        self.decision_history = []
        
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
            
            decision = self.state_machine.step(obs, profile_override=profile_variant)
            
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
                stats = self.learner.stats[(state, profile)]
                if stats.N >= 5:
                    recent_severe_count = sum(1 for s in stats.recent_scores[-5:] if s < -5.0)
                    if recent_severe_count >= 2:
                        # Poison this profile to prevent future selection
                        self.safety_clamp.poison_profile(state, profile)
                        print(f"[HARNESS] Poisoned ({state}, {profile}) due to repeated severe events")
        
        if self.logger:
            self.logger.end_episode(success, failure_reason)

    
    def _ee_target_to_joint_target(self, decision: RFSNDecision) -> np.ndarray:
        """
        Convert end-effector target to joint target using damped least squares IK.
        
        Uses MuJoCo Jacobian and iterative position-based IK with damping.
        """
        q_current = self.data.qpos[:7].copy()
        
        # Get end-effector body ID
        ee_body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "panda_hand")
        
        # Iterative IK with damped least squares
        q_ik = q_current.copy()
        alpha = 0.5  # Step size
        damping = 0.01  # Damping factor for numerical stability
        max_iterations = 10
        pos_tolerance = 0.01  # 1cm
        
        for iteration in range(max_iterations):
            # Update MuJoCo data with current joint positions
            data_temp = mj.MjData(self.model)
            data_temp.qpos[:] = self.data.qpos
            data_temp.qpos[:7] = q_ik
            mj.mj_forward(self.model, data_temp)
            
            # Get current EE position
            ee_pos_current = data_temp.xpos[ee_body_id].copy()
            
            # Compute position error
            pos_error = decision.x_target_pos - ee_pos_current
            
            # Check convergence
            if np.linalg.norm(pos_error) < pos_tolerance:
                break
            
            # Compute Jacobian for end-effector position
            jacp = np.zeros((3, self.model.nv))  # Position Jacobian
            jacr = np.zeros((3, self.model.nv))  # Rotation Jacobian
            mj.mj_jacBodyCom(self.model, data_temp, jacp, jacr, ee_body_id)
            
            # Extract arm joints only (first 7 DOF)
            J = jacp[:, :7]
            
            # Damped least squares: dq = J^T (J J^T + λ²I)^{-1} * dx
            # This is more stable than pseudo-inverse for redundant manipulators
            JJT = J @ J.T
            damping_matrix = damping**2 * np.eye(3)
            dq = J.T @ np.linalg.solve(JJT + damping_matrix, pos_error)
            
            # Update joint angles with step size
            q_ik += alpha * dq
            
            # Clamp to joint limits
            for i in range(7):
                q_min = self.model.jnt_range[i, 0]
                q_max = self.model.jnt_range[i, 1]
                q_ik[i] = np.clip(q_ik[i], q_min, q_max)
        
        return q_ik
    
    def _inverse_dynamics_control(
        self,
        q: np.ndarray,
        qd: np.ndarray,
        q_target: np.ndarray,
        decision: Optional[RFSNDecision]
    ) -> np.ndarray:
        """
        Compute control torques using inverse dynamics.
        
        Uses MuJoCo's mj_inverse to compute required torques.
        """
        # Apply MPC gains (from decision if available)
        if decision:
            # Scale KP based on Q weights
            kp_scale = np.sqrt(decision.Q_diag[:7] / 50.0)  # Normalized to base
            KP_local = self.KP * kp_scale
            
            # Scale KD based on Q velocity weights
            kd_scale = np.sqrt(decision.Q_diag[7:14] / 10.0)
            KD_local = self.KD * kd_scale
        else:
            KP_local = self.KP
            KD_local = self.KD
        
        # PD control
        q_error = q_target - q
        dq_desired = KP_local * q_error
        
        # Create temp data for inverse dynamics
        data_temp = mj.MjData(self.model)
        data_temp.qpos[:] = self.data.qpos
        data_temp.qvel[:] = self.data.qvel
        
        # Set desired acceleration
        qacc_full = np.zeros(self.model.nv)
        qacc_full[:7] = dq_desired - KD_local * qd
        data_temp.qacc[:] = qacc_full
        
        # Compute inverse dynamics
        mj.mj_inverse(self.model, data_temp)
        tau = data_temp.qfrc_inverse[:7].copy()
        
        # Apply torque scale if decision provided
        if decision:
            tau *= decision.max_tau_scale
        
        # Saturate
        tau = np.clip(tau, -87.0, 87.0)
        
        return tau
    
    def _check_grasp_quality(self, obs: ObsPacket) -> dict:
        """
        Check grasp quality based on contacts and gripper state.
        
        Returns:
            {
                'has_contact': bool - whether fingers are in contact with object
                'is_stable': bool - whether grasp is stable (both fingers, low motion)
                'quality': float - grasp quality score 0-1
            }
        """
        result = {
            'has_contact': obs.obj_contact and obs.ee_contact,
            'is_stable': False,
            'quality': 0.0
        }
        
        # Check if both fingers are in contact
        if not result['has_contact']:
            return result
        
        # Check gripper width (closed enough)
        gripper_width = obs.gripper.get('width', 0.0)
        is_closed = gripper_width < 0.06  # Gripper should be mostly closed
        
        # Check relative motion (object should move with EE)
        if obs.x_obj_pos is not None:
            # Object velocity should be small relative to workspace
            obj_vel_norm = np.linalg.norm(obs.xd_ee_lin)
            is_low_motion = obj_vel_norm < 0.1  # Less than 10cm/s
        else:
            is_low_motion = True
        
        # Grasp is stable if closed and low motion
        result['is_stable'] = is_closed and is_low_motion and result['has_contact']
        
        # Compute quality score
        quality = 0.0
        if result['has_contact']:
            quality += 0.4
        if is_closed:
            quality += 0.3
        if is_low_motion:
            quality += 0.3
        
        result['quality'] = quality
        
        return result
    
    def get_stats(self) -> dict:
        tau = data_temp.qfrc_inverse[:7].copy()
        
        # Apply torque scale if decision provided
        if decision:
            tau *= decision.max_tau_scale
        
        # Saturate
        tau = np.clip(tau, -87.0, 87.0)
        
        return tau
    
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

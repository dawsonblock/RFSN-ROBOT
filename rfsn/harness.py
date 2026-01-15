"""
RFSN-MPC Unified Harness
=========================
Main integration point: wraps MPC controller with RFSN executive layer.

Supports 3 modes:
1. MPC only (baseline)
2. RFSN without learning
3. RFSN with learning
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
    Unified harness for MPC + RFSN integration.
    
    Modes:
    - "mpc_only": Baseline MPC with fixed targets
    - "rfsn": RFSN state machine without learning
    - "rfsn_learning": RFSN state machine with safe learning
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
        
        # Gripper control (simple open/close)
        if self.rfsn_enabled and decision:
            if decision.task_mode in ["GRASP", "LIFT", "TRANSPORT"]:
                self.data.ctrl[7] = -50.0  # Close left
                self.data.ctrl[8] = 50.0   # Close right
            else:
                self.data.ctrl[7] = 0.0
                self.data.ctrl[8] = 0.0
        else:
            self.data.ctrl[7] = 0.0
            self.data.ctrl[8] = 0.0
        
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
        """End current episode."""
        self.episode_active = False
        
        # Update learner if enabled
        if self.learner and len(self.obs_history) > 0:
            score, violations = self.learner.compute_score(
                self.obs_history, 
                self.decision_history
            )
            
            # Update stats for each (state, profile) used
            state_profile_visits = {}
            for decision in self.decision_history:
                key = (decision.task_mode, decision.rollback_token.split('_')[1] if '_' in decision.rollback_token else 'base')
                state_profile_visits[key] = state_profile_visits.get(key, 0) + 1
            
            for (state, profile), count in state_profile_visits.items():
                self.learner.update_stats(state, profile, score, violations, self.t)
        
        if self.logger:
            self.logger.end_episode(success, failure_reason)
    
    def _ee_target_to_joint_target(self, decision: RFSNDecision) -> np.ndarray:
        """
        Convert end-effector target to joint target.
        
        This is a simplified IK stub. Real implementation would use:
        - MuJoCo's IK solver
        - Analytical IK for Panda
        - Numerical optimization
        
        For now, we return a reasonable joint configuration.
        """
        # Simplified: use current joint config and adjust based on EE target
        # This is NOT proper IK but allows the system to function
        q_current = self.data.qpos[:7].copy()
        
        # Get current EE position
        ee_body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "panda_hand")
        ee_pos_current = self.data.xpos[ee_body_id].copy()
        
        # Compute error
        pos_error = decision.x_target_pos - ee_pos_current
        
        # Simple Jacobian-based adjustment (very approximate)
        # In practice, use mj_jacBodyCom or proper IK
        q_target = q_current.copy()
        
        # Adjust joint 2 for vertical (Z)
        q_target[1] += pos_error[2] * 0.5
        
        # Adjust joint 1 for rotation around Z
        q_target[0] += np.arctan2(decision.x_target_pos[1], decision.x_target_pos[0]) * 0.3
        
        # Clamp to joint limits
        for i in range(7):
            q_min = self.model.jnt_range[i, 0]
            q_max = self.model.jnt_range[i, 1]
            q_target[i] = np.clip(q_target[i], q_min, q_max)
        
        return q_target
    
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

"""
Receding Horizon MPC for Joint-Space Tracking
==============================================
Real MPC optimizer with warm-start, time budget, and safe fallback.

This module implements a receding-horizon Model Predictive Controller that:
- Solves a finite-horizon optimization problem at each timestep
- Uses simple discrete-time joint-space dynamics
- Applies only the first control step (receding horizon)
- Outputs reference trajectories (q_ref, qd_ref) for the ID controller

The MPC fields (horizon_steps, Q_diag, R_diag, terminal_Q_diag, du_penalty) 
from RFSN profiles now directly control the optimization behavior.
"""

import numpy as np
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class MPCConfig:
    """Configuration for MPC solver."""
    
    # Horizon
    H_min: int = 5
    H_max: int = 30
    
    # Joint acceleration bounds (rad/s^2)
    qdd_min: np.ndarray = field(default_factory=lambda: -2.0 * np.ones(7))
    qdd_max: np.ndarray = field(default_factory=lambda: 2.0 * np.ones(7))
    
    # Joint velocity bounds (rad/s) - soft constraint via penalty
    qd_min: np.ndarray = field(default_factory=lambda: -2.0 * np.ones(7))
    qd_max: np.ndarray = field(default_factory=lambda: 2.0 * np.ones(7))
    
    # Joint position bounds - use model limits (soft penalty)
    q_penalty_margin: float = 0.1  # Penalty starts this close to limits
    
    # Solver parameters
    max_iterations: int = 100
    convergence_tol: float = 1e-4  # Relative cost improvement threshold
    time_budget_ms: float = 50.0  # Maximum solve time per step
    
    # Optimization parameters
    learning_rate: float = 0.1  # Step size for gradient descent
    line_search_steps: int = 3  # Backtracking line search iterations
    gradient_clip: float = 10.0  # Gradient clipping to prevent instability
    
    # Warm-start
    warm_start: bool = True
    
    def __post_init__(self):
        """Validate configuration."""
        # Basic scalar checks
        assert self.H_min > 0 and self.H_max >= self.H_min
        assert self.max_iterations > 0
        assert self.time_budget_ms > 0.0
        assert self.learning_rate > 0.0

        # Joint acceleration bounds: shape (7,) and elementwise qdd_min < qdd_max
        assert isinstance(self.qdd_min, np.ndarray) and isinstance(self.qdd_max, np.ndarray)
        assert self.qdd_min.shape == (7,) and self.qdd_max.shape == (7,)
        assert np.all(self.qdd_min < self.qdd_max)

        # Joint velocity bounds: shape (7,) and elementwise qd_min < qd_max
        assert isinstance(self.qd_min, np.ndarray) and isinstance(self.qd_max, np.ndarray)
        assert self.qd_min.shape == (7,) and self.qd_max.shape == (7,)
        assert np.all(self.qd_min < self.qd_max)
@dataclass
class MPCResult:
    """Result from MPC solve."""
    
    # Status
    converged: bool
    solve_time_ms: float
    iters: int
    reason: str  # "converged", "timeout", "max_iters", "failed"
    
    # Cost breakdown
    cost_total: float
    cost_tracking: float = 0.0
    cost_velocity: float = 0.0
    cost_effort: float = 0.0
    cost_smoothness: float = 0.0
    cost_terminal: float = 0.0
    cost_constraint: float = 0.0
    
    # Output references for inverse dynamics controller
    q_ref_next: Optional[np.ndarray] = None  # shape (7,)
    qd_ref_next: Optional[np.ndarray] = None  # shape (7,)
    qdd_cmd_next: Optional[np.ndarray] = None  # shape (7,) - first control in horizon
    
    # Debug info
    cost_history: list = field(default_factory=list)
    gradient_norm: float = 0.0


class RecedingHorizonMPC:
    """
    Receding Horizon Model Predictive Controller for joint-space tracking.
    
    Dynamics Model (discrete-time):
        x = [q(7), qd(7)]  # State: joint positions and velocities
        u = qdd(7)         # Control: joint accelerations
        
        q_{t+1} = q_t + dt * qd_t
        qd_{t+1} = qd_t + dt * qdd_t
    
    Cost Function:
        J = Σ_{t=0}^{H-1} [
            (q_t - q_target)^T Q (q_t - q_target)          # Position tracking
            + qd_t^T Qd qd_t                                # Velocity penalty (near contact)
            + qdd_t^T R qdd_t                               # Effort penalty
            + du_penalty * ||qdd_t - qdd_{t-1}||^2          # Smoothness
        ] + (q_H - q_target)^T Q_terminal (q_H - q_target) # Terminal cost
    
    Constraints:
        - qdd bounds (hard clip)
        - qd bounds (soft penalty)
        - Joint limit proximity penalty (soft barrier)
    
    Outputs:
        - q_ref_next, qd_ref_next: First step of optimized trajectory
        - These are fed to the existing inverse dynamics PD controller
    """
    
    def __init__(self, config: Optional[MPCConfig] = None):
        """
        Initialize MPC solver.
        
        Args:
            config: MPC configuration (uses defaults if None)
        """
        self.config = config or MPCConfig()
        
        # Warm-start buffer: previous solution shifted by 1 step
        self.prev_qdd_trajectory = None  # shape (H, 7)
        self.prev_horizon = None
        
    def solve(
        self,
        q: np.ndarray,
        qd: np.ndarray,
        q_target: np.ndarray,
        dt: float,
        decision_params: dict,
        joint_limits: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> MPCResult:
        """
        Solve MPC optimization problem.
        
        Args:
            q: Current joint positions (7,)
            qd: Current joint velocities (7,)
            q_target: Target joint positions (7,)
            dt: Timestep
            decision_params: Dictionary with MPC parameters from RFSN decision:
                - horizon_steps: int
                - Q_diag: np.ndarray (14,) [position (7), velocity (7)]
                - R_diag: np.ndarray (7,)
                - terminal_Q_diag: np.ndarray (14,)
                - du_penalty: float
                - joint_limit_proximity: float (0..1, for soft constraint)
            joint_limits: Optional (q_min, q_max) tuple for bounds checking
        
        Returns:
            MPCResult with optimized trajectory and metrics
        """
        t_start = time.perf_counter()
        
        # Extract parameters
        H = np.clip(decision_params['horizon_steps'], self.config.H_min, self.config.H_max)
        Q_pos = decision_params['Q_diag'][:7]  # Position tracking weights
        Q_vel = decision_params['Q_diag'][7:14]  # Velocity penalty weights
        R = decision_params['R_diag']  # Effort weights
        Q_terminal_pos = decision_params['terminal_Q_diag'][:7]
        du_penalty = decision_params['du_penalty']
        
        # Initialize trajectory with warm-start if available
        if self.config.warm_start and self.prev_qdd_trajectory is not None and self.prev_horizon == H:
            # Shift previous solution by 1 step and pad with zeros
            qdd_trajectory = np.vstack([
                self.prev_qdd_trajectory[1:, :],
                np.zeros((1, 7))
            ])
        else:
            # Cold start: zero accelerations
            qdd_trajectory = np.zeros((H, 7))
        
        # Optimization loop
        cost_history = []
        converged = False
        reason = "max_iters"
        
        for iteration in range(self.config.max_iterations):
            # Check time budget
            elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            if elapsed_ms > self.config.time_budget_ms:
                reason = "timeout"
                break
            
            # Forward rollout to get trajectory
            q_traj, qd_traj = self._rollout_dynamics(q, qd, qdd_trajectory, dt, H)
            
            # Compute cost
                improvement = (cost_history[-2] - cost_history[-1]) / (abs(cost_history[-2]) + 1e-8)
                if abs(improvement) < self.config.convergence_tol:
                    converged = True
                joint_limits
            )
            cost_history.append(cost_total)
            
            # Check convergence
            if len(cost_history) > 1:
                improvement = (cost_history[-2] - cost_history[-1]) / (abs(cost_history[-2]) + 1e-8)
                if improvement < self.config.convergence_tol and improvement >= 0:
                    converged = True
                    reason = "converged"
                    break
            
            # Compute gradient
            grad = self._compute_gradient(
                q_traj, qd_traj, qdd_trajectory, q_target, dt, H,
                Q_pos, Q_vel, R, Q_terminal_pos, du_penalty,
                joint_limits
            )
            
            # Clip gradient
            grad_norm = np.linalg.norm(grad)
            if grad_norm > self.config.gradient_clip:
                grad = grad * (self.config.gradient_clip / grad_norm)
            
            # Line search for step size
            best_qdd = qdd_trajectory.copy()
            best_cost = cost_total
            step_size = self.config.learning_rate
            
            for _ in range(self.config.line_search_steps):
                # Gradient descent step
                qdd_new = qdd_trajectory - step_size * grad
                
                # Project onto constraints (hard clip on qdd bounds)
                qdd_new = np.clip(qdd_new, self.config.qdd_min, self.config.qdd_max)
                
                # Evaluate new trajectory
                q_new, qd_new = self._rollout_dynamics(q, qd, qdd_new, dt, H)
                cost_new, _ = self._compute_cost(
                    q_new, qd_new, qdd_new, q_target,
                    Q_pos, Q_vel, R, Q_terminal_pos, du_penalty,
                    joint_limits
                )
                
                if cost_new < best_cost:
                    best_cost = cost_new
                    best_qdd = qdd_new
                    break
                
                # Reduce step size
                step_size *= 0.5
            
            # Update trajectory
            qdd_trajectory = best_qdd
        
        # Final rollout
        q_traj, qd_traj = self._rollout_dynamics(q, qd, qdd_trajectory, dt, H)
        final_cost, cost_breakdown = self._compute_cost(
            q_traj, qd_traj, qdd_trajectory, q_target,
            Q_pos, Q_vel, R, Q_terminal_pos, du_penalty,
            joint_limits
        )
        
        # Extract first step (receding horizon)
        q_ref_next = q_traj[1, :]  # Next state after applying first control
        qd_ref_next = qd_traj[1, :]
        qdd_cmd_next = qdd_trajectory[0, :]
        
        # Store for warm-start
        self.prev_qdd_trajectory = qdd_trajectory
        self.prev_horizon = H
        
        # Compute solve time
        solve_time_ms = (time.perf_counter() - t_start) * 1000.0
        
        return MPCResult(
            converged=converged,
            solve_time_ms=solve_time_ms,
            iters=iteration + 1,
            reason=reason,
            cost_total=final_cost,
            cost_tracking=cost_breakdown['tracking'],
            cost_velocity=cost_breakdown['velocity'],
            cost_effort=cost_breakdown['effort'],
            cost_smoothness=cost_breakdown['smoothness'],
            cost_terminal=cost_breakdown['terminal'],
            cost_constraint=cost_breakdown['constraint'],
            q_ref_next=q_ref_next,
            qd_ref_next=qd_ref_next,
            qdd_cmd_next=qdd_cmd_next,
            cost_history=cost_history,
            gradient_norm=grad_norm if 'grad_norm' in locals() else 0.0
        )
    
    def _rollout_dynamics(
        self,
        q0: np.ndarray,
        qd0: np.ndarray,
        qdd_trajectory: np.ndarray,
        dt: float,
        H: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward rollout of discrete-time dynamics.
        
        Args:
            q0: Initial joint positions (7,)
            qd0: Initial joint velocities (7,)
            qdd_trajectory: Control trajectory (H, 7)
            dt: Timestep
            H: Horizon length
        
        Returns:
            q_traj: Joint positions (H+1, 7) - includes initial state
            qd_traj: Joint velocities (H+1, 7)
        """
        q_traj = np.zeros((H + 1, 7))
        qd_traj = np.zeros((H + 1, 7))
        
        q_traj[0, :] = q0
        qd_traj[0, :] = qd0
        
        for t in range(H):
            # Semi-implicit (symplectic) Euler integration:
            # 1) update velocity from acceleration
            qd_traj[t + 1, :] = qd_traj[t, :] + dt * qdd_trajectory[t, :]
            # 2) update position using the new velocity
            q_traj[t + 1, :] = q_traj[t, :] + dt * qd_traj[t + 1, :]
        
        return q_traj, qd_traj
    
    def _compute_cost(
        self,
        q_traj: np.ndarray,
        qd_traj: np.ndarray,
        qdd_trajectory: np.ndarray,
        q_target: np.ndarray,
        Q_pos: np.ndarray,
        Q_vel: np.ndarray,
        R: np.ndarray,
        Q_terminal_pos: np.ndarray,
        du_penalty: float,
        joint_limits: Optional[Tuple[np.ndarray, np.ndarray]]
    ) -> Tuple[float, dict]:
        """
        Compute total cost and breakdown.
        
        Returns:
            total_cost, cost_breakdown_dict
        """
        H = qdd_trajectory.shape[0]
        
        # Tracking cost: position error
        q_error = q_traj[:-1, :] - q_target  # (H, 7)
        cost_tracking = np.sum(Q_pos * q_error**2)
        
        # Velocity penalty (useful near contact states)
        cost_velocity = np.sum(Q_vel * qd_traj[:-1, :]**2)
        
        # Effort cost
        cost_effort = np.sum(R * qdd_trajectory**2)
        
        # Smoothness cost (acceleration changes)
        cost_smoothness = 0.0
        if H > 1:
            qdd_diff = qdd_trajectory[1:, :] - qdd_trajectory[:-1, :]
            cost_smoothness = du_penalty * np.sum(qdd_diff**2)
        
        # Terminal cost
        q_terminal_error = q_traj[-1, :] - q_target
        cost_terminal = np.sum(Q_terminal_pos * q_terminal_error**2)
        
        # Constraint penalties
        cost_constraint = 0.0
        
        # Soft velocity bounds
        qd_violations_upper = np.maximum(0, qd_traj - self.config.qd_max)
        qd_violations_lower = np.maximum(0, self.config.qd_min - qd_traj)
        cost_constraint += 100.0 * (np.sum(qd_violations_upper**2) + np.sum(qd_violations_lower**2))
        
        # Joint limit proximity penalty (soft barrier)
        if joint_limits is not None:
            q_min, q_max = joint_limits
            margin = self.config.q_penalty_margin
            
            # Penalty near lower limit
            dist_to_lower = q_traj - q_min
            near_lower = dist_to_lower < margin
            if np.any(near_lower):
                penalty_lower = np.maximum(0, margin - dist_to_lower[near_lower])
                cost_constraint += 100.0 * np.sum(penalty_lower**2)
            
            # Penalty near upper limit
            dist_to_upper = q_max - q_traj
            near_upper = dist_to_upper < margin
            if np.any(near_upper):
                penalty_upper = np.maximum(0, margin - dist_to_upper[near_upper])
                cost_constraint += 100.0 * np.sum(penalty_upper**2)
        
        total_cost = (cost_tracking + cost_velocity + cost_effort + 
                     cost_smoothness + cost_terminal + cost_constraint)
        
        breakdown = {
            'tracking': cost_tracking,
            'velocity': cost_velocity,
            'effort': cost_effort,
            'smoothness': cost_smoothness,
            'terminal': cost_terminal,
            'constraint': cost_constraint
        }
        
        return total_cost, breakdown
    
    def _compute_gradient(
        self,
        q_traj: np.ndarray,
        qd_traj: np.ndarray,
        qdd_trajectory: np.ndarray,
        q_target: np.ndarray,
        dt: float,
        H: int,
        Q_pos: np.ndarray,
        Q_vel: np.ndarray,
        R: np.ndarray,
        Q_terminal_pos: np.ndarray,
        du_penalty: float,
        joint_limits: Optional[Tuple[np.ndarray, np.ndarray]]
    ) -> np.ndarray:
        """
        Compute gradient of cost w.r.t. control trajectory using finite differences.
        
        This is a simplified gradient computation that approximates the true gradient.
        For a full implementation, you would use backpropagation through time or
        adjoint methods, but finite differences are sufficient for this application.
        
        Returns:
            gradient: (H, 7) array
        """
        grad = np.zeros((H, 7))
        epsilon = 1e-6
        
        # Base cost
        base_cost, _ = self._compute_cost(
            q_traj, qd_traj, qdd_trajectory, q_target,
            Q_pos, Q_vel, R, Q_terminal_pos, du_penalty,
            joint_limits
        )
        
        # Finite difference for each control variable
        for t in range(H):
            for i in range(7):
                # Perturb control
                qdd_perturbed = qdd_trajectory.copy()
                qdd_perturbed[t, i] += epsilon
                
                # Rollout and evaluate
                q_new, qd_new = self._rollout_dynamics(
                    q_traj[0, :], qd_traj[0, :], qdd_perturbed, dt, H
                )
                cost_new, _ = self._compute_cost(
                    q_new, qd_new, qdd_perturbed, q_target,
                    Q_pos, Q_vel, R, Q_terminal_pos, du_penalty,
                    joint_limits
                )
                
                # Finite difference
                grad[t, i] = (cost_new - base_cost) / epsilon
        
        return grad
    
    def reset_warm_start(self):
        """Reset warm-start buffer (e.g., at episode start)."""
        self.prev_qdd_trajectory = None
        self.prev_horizon = None

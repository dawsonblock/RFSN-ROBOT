"""
RFSN Logger: Episode and event logging
=======================================
Logs observations, decisions, and events to JSONL and CSV.
"""

import json
import csv
import os
from datetime import datetime
from typing import List, Dict, Any
from .obs_packet import ObsPacket
from .decision import RFSNDecision


class RFSNLogger:
    """Logger for RFSN episodes and events."""
    
    def __init__(self, run_dir: str = None):
        """
        Initialize logger with run directory.
        
        Args:
            run_dir: Directory for this run (default: runs/<timestamp>)
        """
        if run_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = f"runs/{timestamp}"
        
        self.run_dir = run_dir
        os.makedirs(self.run_dir, exist_ok=True)
        
        self.episodes_csv_path = os.path.join(self.run_dir, "episodes.csv")
        self.events_jsonl_path = os.path.join(self.run_dir, "events.jsonl")
        
        # Initialize CSV
        self._init_episodes_csv()
        
        # Episode tracking
        self.current_episode = None
        self.episode_count = 0
        
        print(f"[LOGGER] Logging to: {self.run_dir}")
    
    def _init_episodes_csv(self):
        """Initialize episodes CSV with headers."""
        with open(self.episodes_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'episode_id',
                'task_name',
                'success',
                'failure_reason',
                'duration_s',
                'num_steps',
                'collision_count',
                'self_collision_count',
                'table_collision_count',
                'torque_sat_count',
                'mpc_fail_count',
                'mean_mpc_solve_ms',
                'max_penetration',
                'max_joint_limit_prox',
                'energy_proxy',
                'smoothness_proxy',
                'final_distance_to_goal',
                'initial_cube_x',
                'initial_cube_y',
                'initial_cube_z',
                'goal_x',
                'goal_y',
                'goal_z',
                'recover_time_steps',
            ])
    
    def start_episode(self, episode_id: int, task_name: str, 
                     initial_cube_pos: list = None, goal_pos: list = None):
        """Start logging a new episode."""
        self.current_episode = {
            'episode_id': episode_id,
            'task_name': task_name,
            'start_time': None,
            'obs_history': [],
            'decision_history': [],
            'events': [],
            'initial_cube_pos': initial_cube_pos,
            'goal_pos': goal_pos,
        }
        self.episode_count += 1
    
    def log_step(self, obs: ObsPacket, decision: RFSNDecision):
        """Log a single control step."""
        if self.current_episode is None:
            return
        
        if self.current_episode['start_time'] is None:
            self.current_episode['start_time'] = obs.t
        
        self.current_episode['obs_history'].append(obs)
        self.current_episode['decision_history'].append(decision)
        
        # Log events
        if obs.self_collision:
            self._log_event('self_collision', obs.t, {'state': decision.task_mode})
        if obs.table_collision:
            self._log_event('table_collision', obs.t, {'state': decision.task_mode})
        if obs.torque_sat_count > 0:
            self._log_event('torque_saturation', obs.t, 
                          {'count': obs.torque_sat_count, 'state': decision.task_mode})
        if not obs.mpc_converged:
            self._log_event('mpc_nonconvergence', obs.t, {'state': decision.task_mode})
    
    def _log_event(self, event_type: str, time: float, data: dict):
        """Log an event to JSONL."""
        # Convert numpy types to native Python types
        converted_data = {}
        for key, value in data.items():
            if isinstance(value, np.integer):
                converted_data[key] = int(value)
            elif isinstance(value, np.floating):
                converted_data[key] = float(value)
            elif isinstance(value, np.ndarray):
                converted_data[key] = value.tolist()
            else:
                converted_data[key] = value
        
        event = {
            'episode_id': self.current_episode['episode_id'],
            'event_type': event_type,
            'time': float(time),
            'data': converted_data,
        }
        
        with open(self.events_jsonl_path, 'a') as f:
            f.write(json.dumps(event) + '\n')
        
        self.current_episode['events'].append(event)
    
    def end_episode(self, success: bool = False, failure_reason: str = None):
        """End current episode and write summary."""
        if self.current_episode is None:
            return
        
        obs_history = self.current_episode['obs_history']
        decision_history = self.current_episode['decision_history']
        
        if not obs_history:
            return
        
        # Compute episode statistics
        duration = obs_history[-1].t - self.current_episode['start_time']
        num_steps = len(obs_history)
        
        collision_count = sum(1 for o in obs_history if o.self_collision or o.table_collision)
        self_collision_count = sum(1 for o in obs_history if o.self_collision)
        table_collision_count = sum(1 for o in obs_history if o.table_collision)
        torque_sat_count = sum(o.torque_sat_count for o in obs_history)
        mpc_fail_count = sum(1 for o in obs_history if not o.mpc_converged)
        
        solve_times = [o.mpc_solve_time_ms for o in obs_history if o.mpc_solve_time_ms > 0]
        mean_mpc_solve = sum(solve_times) / len(solve_times) if solve_times else 0.0
        
        max_penetration = max(o.penetration for o in obs_history)
        max_joint_limit_prox = max(o.joint_limit_proximity for o in obs_history)
        
        # Energy and smoothness proxies (would need torque history)
        energy_proxy = 0.0  # Placeholder
        smoothness_proxy = 0.0  # Placeholder
        
        # Distance to goal (if available)
        final_obs = obs_history[-1]
        if final_obs.x_goal_pos is not None:
            final_distance = float(np.linalg.norm(final_obs.x_ee_pos - final_obs.x_goal_pos))
        else:
            final_distance = 0.0
        
        # Extract initial cube and goal positions
        initial_cube_pos = self.current_episode.get('initial_cube_pos')
        goal_pos = self.current_episode.get('goal_pos')
        
        # Count RECOVER time
        recover_time_steps = 0
        for event in self.current_episode['events']:
            if event['event_type'] == 'state_change' and event.get('data', {}).get('new_state') == 'RECOVER':
                recover_time_steps += 1
        
        # Write to CSV
        with open(self.episodes_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                self.current_episode['episode_id'],
                self.current_episode['task_name'],
                success,
                failure_reason or '',
                duration,
                num_steps,
                collision_count,
                self_collision_count,
                table_collision_count,
                torque_sat_count,
                mpc_fail_count,
                mean_mpc_solve,
                max_penetration,
                max_joint_limit_prox,
                energy_proxy,
                smoothness_proxy,
                final_distance,
                initial_cube_pos[0] if initial_cube_pos else 0.0,
                initial_cube_pos[1] if initial_cube_pos else 0.0,
                initial_cube_pos[2] if initial_cube_pos else 0.0,
                goal_pos[0] if goal_pos else 0.0,
                goal_pos[1] if goal_pos else 0.0,
                goal_pos[2] if goal_pos else 0.0,
                recover_time_steps,
            ])
        
        # Log episode end event
        self._log_event('episode_end', obs_history[-1].t, {
            'success': success,
            'failure_reason': failure_reason,
            'duration': duration,
        })
        
        print(f"[LOGGER] Episode {self.current_episode['episode_id']} complete: "
              f"success={success}, duration={duration:.2f}s, steps={num_steps}")
        
        self.current_episode = None
    
    def get_run_dir(self) -> str:
        """Get the run directory path."""
        return self.run_dir


# Import numpy for distance calculation
import numpy as np

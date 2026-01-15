"""
Profile Library: Safe parameter profiles per state
===================================================
Defines 3-5 variants per state (base, precise, smooth, fast, stable).
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class MPCProfile:
    """MPC parameter profile."""
    name: str
    horizon_steps: int
    Q_diag: np.ndarray
    R_diag: np.ndarray
    terminal_Q_diag: np.ndarray
    du_penalty: float
    max_tau_scale: float = 1.0
    contact_policy: str = "AVOID"
    
    def to_dict(self):
        return {
            'name': self.name,
            'horizon_steps': self.horizon_steps,
            'Q_diag': self.Q_diag.tolist(),
            'R_diag': self.R_diag.tolist(),
            'terminal_Q_diag': self.terminal_Q_diag.tolist(),
            'du_penalty': self.du_penalty,
            'max_tau_scale': self.max_tau_scale,
            'contact_policy': self.contact_policy,
        }


class ProfileLibrary:
    """Library of safe MPC profiles per state."""
    
    def __init__(self):
        """Initialize profile library with safe defaults."""
        self.profiles = self._create_profiles()
        
    def _create_profiles(self) -> Dict[str, Dict[str, MPCProfile]]:
        """Create profile variants for each state."""
        profiles = {}
        
        # Common base values
        base_Q = np.array([50.0] * 7 + [10.0] * 7)  # [pos, vel]
        base_R = 0.01 * np.ones(7)
        base_terminal_Q = 10.0 * base_Q
        
        # IDLE state
        profiles["IDLE"] = {
            "base": MPCProfile(
                name="idle_base",
                horizon_steps=10,
                Q_diag=base_Q.copy(),
                R_diag=base_R.copy(),
                terminal_Q_diag=base_terminal_Q.copy(),
                du_penalty=0.01,
                max_tau_scale=0.5,  # Conservative
                contact_policy="AVOID"
            )
        }
        
        # REACH_PREGRASP state
        profiles["REACH_PREGRASP"] = {
            "base": MPCProfile(
                name="reach_pregrasp_base",
                horizon_steps=15,
                Q_diag=np.array([100.0] * 7 + [20.0] * 7),
                R_diag=0.01 * np.ones(7),
                terminal_Q_diag=np.array([200.0] * 7 + [40.0] * 7),
                du_penalty=0.01,
                max_tau_scale=0.8,
                contact_policy="AVOID"
            ),
            "precise": MPCProfile(
                name="reach_pregrasp_precise",
                horizon_steps=20,
                Q_diag=np.array([200.0] * 7 + [30.0] * 7),  # Higher tracking
                R_diag=0.01 * np.ones(7),
                terminal_Q_diag=np.array([400.0] * 7 + [60.0] * 7),
                du_penalty=0.01,
                max_tau_scale=0.8,
                contact_policy="AVOID"
            ),
            "smooth": MPCProfile(
                name="reach_pregrasp_smooth",
                horizon_steps=15,
                Q_diag=np.array([80.0] * 7 + [15.0] * 7),
                R_diag=0.05 * np.ones(7),  # Higher R
                terminal_Q_diag=np.array([160.0] * 7 + [30.0] * 7),
                du_penalty=0.05,  # Higher du penalty
                max_tau_scale=0.7,
                contact_policy="AVOID"
            ),
            "fast": MPCProfile(
                name="reach_pregrasp_fast",
                horizon_steps=8,  # Shorter horizon
                Q_diag=np.array([120.0] * 7 + [25.0] * 7),
                R_diag=0.005 * np.ones(7),  # Lower R for responsiveness
                terminal_Q_diag=np.array([240.0] * 7 + [50.0] * 7),
                du_penalty=0.005,
                max_tau_scale=0.9,
                contact_policy="AVOID"
            ),
            "stable": MPCProfile(
                name="reach_pregrasp_stable",
                horizon_steps=12,
                Q_diag=np.array([60.0] * 7 + [12.0] * 7),  # Lower gains
                R_diag=0.02 * np.ones(7),
                terminal_Q_diag=np.array([120.0] * 7 + [24.0] * 7),
                du_penalty=0.02,
                max_tau_scale=0.6,  # Very conservative
                contact_policy="AVOID"
            ),
        }
        
        # REACH_GRASP state (similar to REACH_PREGRASP but allows EE contact)
        profiles["REACH_GRASP"] = {
            "base": MPCProfile(
                name="reach_grasp_base",
                horizon_steps=12,
                Q_diag=np.array([100.0] * 7 + [20.0] * 7),
                R_diag=0.01 * np.ones(7),
                terminal_Q_diag=np.array([200.0] * 7 + [40.0] * 7),
                du_penalty=0.01,
                max_tau_scale=0.7,
                contact_policy="ALLOW_EE"
            ),
            "precise": MPCProfile(
                name="reach_grasp_precise",
                horizon_steps=15,
                Q_diag=np.array([150.0] * 7 + [25.0] * 7),
                R_diag=0.01 * np.ones(7),
                terminal_Q_diag=np.array([300.0] * 7 + [50.0] * 7),
                du_penalty=0.01,
                max_tau_scale=0.7,
                contact_policy="ALLOW_EE"
            ),
            "smooth": MPCProfile(
                name="reach_grasp_smooth",
                horizon_steps=12,
                Q_diag=np.array([80.0] * 7 + [15.0] * 7),
                R_diag=0.05 * np.ones(7),
                terminal_Q_diag=np.array([160.0] * 7 + [30.0] * 7),
                du_penalty=0.05,
                max_tau_scale=0.6,
                contact_policy="ALLOW_EE"
            ),
        }
        
        # GRASP state (closing gripper, holding position)
        profiles["GRASP"] = {
            "base": MPCProfile(
                name="grasp_base",
                horizon_steps=10,
                Q_diag=np.array([150.0] * 7 + [30.0] * 7),  # Hold position tight
                R_diag=0.02 * np.ones(7),
                terminal_Q_diag=np.array([300.0] * 7 + [60.0] * 7),
                du_penalty=0.02,
                max_tau_scale=0.6,
                contact_policy="ALLOW_EE"
            ),
            "stable": MPCProfile(
                name="grasp_stable",
                horizon_steps=10,
                Q_diag=np.array([100.0] * 7 + [20.0] * 7),
                R_diag=0.03 * np.ones(7),
                terminal_Q_diag=np.array([200.0] * 7 + [40.0] * 7),
                du_penalty=0.03,
                max_tau_scale=0.5,
                contact_policy="ALLOW_EE"
            ),
        }
        
        # LIFT state
        profiles["LIFT"] = {
            "base": MPCProfile(
                name="lift_base",
                horizon_steps=15,
                Q_diag=np.array([120.0] * 7 + [25.0] * 7),
                R_diag=0.015 * np.ones(7),
                terminal_Q_diag=np.array([240.0] * 7 + [50.0] * 7),
                du_penalty=0.015,
                max_tau_scale=0.8,
                contact_policy="ALLOW_EE"
            ),
            "smooth": MPCProfile(
                name="lift_smooth",
                horizon_steps=18,
                Q_diag=np.array([100.0] * 7 + [20.0] * 7),
                R_diag=0.03 * np.ones(7),
                terminal_Q_diag=np.array([200.0] * 7 + [40.0] * 7),
                du_penalty=0.03,
                max_tau_scale=0.7,
                contact_policy="ALLOW_EE"
            ),
            "fast": MPCProfile(
                name="lift_fast",
                horizon_steps=10,
                Q_diag=np.array([140.0] * 7 + [28.0] * 7),
                R_diag=0.008 * np.ones(7),
                terminal_Q_diag=np.array([280.0] * 7 + [56.0] * 7),
                du_penalty=0.008,
                max_tau_scale=0.9,
                contact_policy="ALLOW_EE"
            ),
        }
        
        # TRANSPORT state
        profiles["TRANSPORT"] = {
            "base": MPCProfile(
                name="transport_base",
                horizon_steps=15,
                Q_diag=np.array([100.0] * 7 + [20.0] * 7),
                R_diag=0.01 * np.ones(7),
                terminal_Q_diag=np.array([200.0] * 7 + [40.0] * 7),
                du_penalty=0.015,
                max_tau_scale=0.8,
                contact_policy="ALLOW_EE"
            ),
            "smooth": MPCProfile(
                name="transport_smooth",
                horizon_steps=20,
                Q_diag=np.array([80.0] * 7 + [16.0] * 7),
                R_diag=0.03 * np.ones(7),
                terminal_Q_diag=np.array([160.0] * 7 + [32.0] * 7),
                du_penalty=0.03,
                max_tau_scale=0.7,
                contact_policy="ALLOW_EE"
            ),
            "fast": MPCProfile(
                name="transport_fast",
                horizon_steps=10,
                Q_diag=np.array([120.0] * 7 + [24.0] * 7),
                R_diag=0.005 * np.ones(7),
                terminal_Q_diag=np.array([240.0] * 7 + [48.0] * 7),
                du_penalty=0.01,
                max_tau_scale=0.85,
                contact_policy="ALLOW_EE"
            ),
        }
        
        # PLACE state
        profiles["PLACE"] = {
            "base": MPCProfile(
                name="place_base",
                horizon_steps=12,
                Q_diag=np.array([120.0] * 7 + [24.0] * 7),
                R_diag=0.02 * np.ones(7),
                terminal_Q_diag=np.array([240.0] * 7 + [48.0] * 7),
                du_penalty=0.02,
                max_tau_scale=0.7,
                contact_policy="ALLOW_PUSH"
            ),
            "smooth": MPCProfile(
                name="place_smooth",
                horizon_steps=15,
                Q_diag=np.array([100.0] * 7 + [20.0] * 7),
                R_diag=0.04 * np.ones(7),
                terminal_Q_diag=np.array([200.0] * 7 + [40.0] * 7),
                du_penalty=0.04,
                max_tau_scale=0.6,
                contact_policy="ALLOW_PUSH"
            ),
        }
        
        # THROW_PREP state
        profiles["THROW_PREP"] = {
            "base": MPCProfile(
                name="throw_prep_base",
                horizon_steps=12,
                Q_diag=np.array([100.0] * 7 + [20.0] * 7),
                R_diag=0.01 * np.ones(7),
                terminal_Q_diag=np.array([200.0] * 7 + [40.0] * 7),
                du_penalty=0.01,
                max_tau_scale=0.8,
                contact_policy="AVOID"
            ),
            "stable": MPCProfile(
                name="throw_prep_stable",
                horizon_steps=15,
                Q_diag=np.array([80.0] * 7 + [16.0] * 7),
                R_diag=0.02 * np.ones(7),
                terminal_Q_diag=np.array([160.0] * 7 + [32.0] * 7),
                du_penalty=0.02,
                max_tau_scale=0.7,
                contact_policy="AVOID"
            ),
        }
        
        # THROW_EXEC state (high speed, low horizon)
        profiles["THROW_EXEC"] = {
            "low_arc": MPCProfile(
                name="throw_low",
                horizon_steps=5,
                Q_diag=np.array([150.0] * 7 + [30.0] * 7),
                R_diag=0.003 * np.ones(7),  # Very responsive
                terminal_Q_diag=np.array([300.0] * 7 + [60.0] * 7),
                du_penalty=0.003,
                max_tau_scale=1.0,  # Full power
                contact_policy="AVOID"
            ),
            "mid_arc": MPCProfile(
                name="throw_mid",
                horizon_steps=6,
                Q_diag=np.array([140.0] * 7 + [28.0] * 7),
                R_diag=0.003 * np.ones(7),
                terminal_Q_diag=np.array([280.0] * 7 + [56.0] * 7),
                du_penalty=0.004,
                max_tau_scale=1.0,
                contact_policy="AVOID"
            ),
            "high_arc": MPCProfile(
                name="throw_high",
                horizon_steps=7,
                Q_diag=np.array([130.0] * 7 + [26.0] * 7),
                R_diag=0.004 * np.ones(7),
                terminal_Q_diag=np.array([260.0] * 7 + [52.0] * 7),
                du_penalty=0.005,
                max_tau_scale=1.0,
                contact_policy="AVOID"
            ),
        }
        
        # RECOVER state (conservative, safe retreat)
        profiles["RECOVER"] = {
            "base": MPCProfile(
                name="recover_base",
                horizon_steps=10,
                Q_diag=np.array([60.0] * 7 + [12.0] * 7),  # Low gains
                R_diag=0.05 * np.ones(7),  # High damping
                terminal_Q_diag=np.array([120.0] * 7 + [24.0] * 7),
                du_penalty=0.05,
                max_tau_scale=0.4,  # Very conservative
                contact_policy="AVOID"
            ),
        }
        
        # FAIL state (minimal motion)
        profiles["FAIL"] = {
            "base": MPCProfile(
                name="fail_base",
                horizon_steps=5,
                Q_diag=np.array([50.0] * 7 + [10.0] * 7),
                R_diag=0.1 * np.ones(7),
                terminal_Q_diag=np.array([100.0] * 7 + [20.0] * 7),
                du_penalty=0.1,
                max_tau_scale=0.3,
                contact_policy="AVOID"
            ),
        }
        
        return profiles
    
    def get_profile(self, state: str, variant: str = "base") -> MPCProfile:
        """Get a specific profile variant for a state."""
        if state not in self.profiles:
            raise ValueError(f"Unknown state: {state}")
        if variant not in self.profiles[state]:
            # Fallback to base if variant doesn't exist
            return self.profiles[state]["base"]
        return self.profiles[state][variant]
    
    def get_variants(self, state: str) -> List[str]:
        """Get all available variants for a state."""
        if state not in self.profiles:
            return []
        return list(self.profiles[state].keys())
    
    def list_states(self) -> List[str]:
        """List all states with profiles."""
        return list(self.profiles.keys())

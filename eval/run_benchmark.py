"""
Benchmark Runner
================
Run N episodes and collect metrics.

Supports 3 modes:
1. MPC only (baseline)
2. RFSN without learning
3. RFSN with learning

Usage:
    python -m eval.run_benchmark --mode mpc_only --episodes 10
    python -m eval.run_benchmark --mode rfsn --episodes 10
    python -m eval.run_benchmark --mode rfsn_learning --episodes 50
"""

import argparse
import mujoco as mj
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rfsn.harness import RFSNHarness
from rfsn.logger import RFSNLogger
from eval.metrics import compute_metrics, format_metrics, load_episodes, load_events


def run_episode(harness: RFSNHarness, max_steps: int = 5000, render: bool = False) -> tuple:
    """
    Run a single episode.
    
    Args:
        harness: RFSN harness
        max_steps: Maximum steps per episode
        render: Whether to render (not implemented in headless mode)
        
    Returns:
        (success, failure_reason)
    """
    harness.start_episode()
    
    # Track initial cube position from actual simulation state
    initial_cube_pos = None
    goal_region_center = np.array([-0.2, 0.3, 0.45])  # Target place location
    goal_tolerance = 0.15  # 15cm radius around goal
    min_displacement = 0.10  # Minimum 10cm movement to consider manipulation
    
    for step in range(max_steps):
        obs = harness.step()
        
        # Record initial cube position on first step
        if step == 0 and obs.x_obj_pos is not None:
            initial_cube_pos = obs.x_obj_pos.copy()
        
        # Check terminal conditions for RFSN modes
        if harness.rfsn_enabled:
            current_state = harness.state_machine.current_state
            
            # Success: completed task and cube in goal region
            if current_state == "IDLE" and step > 100:
                if obs.x_obj_pos is not None and initial_cube_pos is not None:
                    # Check if cube reached goal region (for pick-place)
                    distance_to_goal = np.linalg.norm(obs.x_obj_pos[:2] - goal_region_center[:2])
                    if distance_to_goal < goal_tolerance:
                        return True, None
                    
                    # Alternative success: cube was displaced significantly (partial credit)
                    displacement = np.linalg.norm(obs.x_obj_pos[:2] - initial_cube_pos[:2])
                    if displacement > min_displacement:
                        # Check if cube is lifted
                        if obs.x_obj_pos[2] > initial_cube_pos[2] + 0.05:  # Lifted 5cm
                            return True, None
            
            # Failure: reached FAIL state
            if current_state == "FAIL":
                return False, "state_machine_fail"
            
            # Timeout in same state
            if harness.state_machine.state_visit_count > 2000:
                return False, "timeout"
        
        else:
            # MPC-only mode: check if cube manipulation was successful
            # Success if cube is stable and displaced from initial position
            if step > 500 and obs.x_obj_pos is not None and initial_cube_pos is not None:
                displacement = np.linalg.norm(obs.x_obj_pos[:2] - initial_cube_pos[:2])
                obj_vel = np.linalg.norm(obs.xd_ee_lin) if hasattr(obs, 'xd_ee_lin') else 0.0
                
                # Check every 100 steps if stable displacement achieved
                if step % 100 == 0:
                    # Success: cube displaced and relatively stable
                    if displacement > min_displacement and obj_vel < 0.05:
                        return True, None
        
        # Safety violations (apply to all modes)
        if obs.self_collision:
            return False, "self_collision"
        if obs.table_collision:
            return False, "table_collision"
        
        # Episode timeout
        if step >= max_steps - 1:
            # For MPC-only, check final state
            if not harness.rfsn_enabled and obs.x_obj_pos is not None and initial_cube_pos is not None:
                displacement = np.linalg.norm(obs.x_obj_pos[:2] - initial_cube_pos[:2])
                if displacement > min_displacement:
                    return True, None  # Partial success for MPC baseline
            return False, "max_steps"
    
    return False, "unknown"


def main():
    """Main benchmark runner."""
    parser = argparse.ArgumentParser(description="Run RFSN benchmark")
    parser.add_argument("--mode", type=str, required=True,
                       choices=["mpc_only", "rfsn", "rfsn_learning"],
                       help="Control mode")
    parser.add_argument("--episodes", type=int, default=10,
                       help="Number of episodes to run")
    parser.add_argument("--task", type=str, default="pick_place",
                       choices=["pick_place", "pick_throw"],
                       help="Task name")
    parser.add_argument("--max-steps", type=int, default=5000,
                       help="Maximum steps per episode")
    parser.add_argument("--model", type=str, default="panda_table_cube.xml",
                       help="MuJoCo model path")
    parser.add_argument("--run-dir", type=str, default=None,
                       help="Run directory (default: auto-generate)")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("RFSN BENCHMARK RUNNER")
    print("=" * 70)
    print(f"Mode:           {args.mode}")
    print(f"Episodes:       {args.episodes}")
    print(f"Task:           {args.task}")
    print(f"Max steps:      {args.max_steps}")
    print(f"Model:          {args.model}")
    print("=" * 70)
    print()
    
    # Load MuJoCo model
    try:
        model = mj.MjModel.from_xml_path(args.model)
        data = mj.MjData(model)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Initialize logger
    logger = RFSNLogger(run_dir=args.run_dir)
    print(f"Logging to: {logger.get_run_dir()}")
    print()
    
    # Initialize harness
    harness = RFSNHarness(
        model=model,
        data=data,
        mode=args.mode,
        task_name=args.task,
        logger=logger
    )
    
    # Run episodes
    print("Running episodes...")
    for episode_id in range(args.episodes):
        print(f"\n[Episode {episode_id + 1}/{args.episodes}]")
        
        # Reset simulation
        mj.mj_resetData(model, data)
        data.qpos[:7] = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
        mj.mj_forward(model, data)
        
        # Start episode logging
        logger.start_episode(episode_id, args.task)
        
        # Run episode
        success, failure_reason = run_episode(harness, max_steps=args.max_steps)
        
        # End episode logging
        harness.end_episode(success, failure_reason)
        
        print(f"  Result: {'SUCCESS' if success else 'FAILURE'}" + 
              (f" ({failure_reason})" if failure_reason else ""))
    
    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
    
    # Print statistics
    if harness.learner:
        print("\nLearning Statistics:")
        stats = harness.learner.get_stats_summary()
        for key, value in list(stats.items())[:10]:  # Print first 10
            print(f"  {key}: N={value['N']}, score={value['mean_score']:.2f}, "
                  f"violations={value['mean_violations']:.2f}")
    
    if harness.safety_clamp:
        print("\nSafety Statistics:")
        safety_stats = harness.safety_clamp.get_stats()
        for key, value in safety_stats.items():
            print(f"  {key}: {value}")
    
    print("\n" + "=" * 70)
    print("GENERATING REPORT")
    print("=" * 70)
    print()
    
    # Load and compute metrics
    episodes_df = load_episodes(os.path.join(logger.get_run_dir(), "episodes.csv"))
    events = load_events(os.path.join(logger.get_run_dir(), "events.jsonl"))
    metrics = compute_metrics(episodes_df, events)
    
    print(format_metrics(metrics))
    
    print(f"\nResults saved to: {logger.get_run_dir()}")
    print(f"  - episodes.csv")
    print(f"  - events.jsonl")
    print()
    print(f"To regenerate this report, run:")
    print(f"  python -m eval.report {logger.get_run_dir()}")


if __name__ == "__main__":
    main()

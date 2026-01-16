"""
Evaluation Metrics
==================
Compute success rate, collision rate, MPC stats, etc.
"""

import pandas as pd
import json
from typing import Dict, List


def load_episodes(csv_path: str) -> pd.DataFrame:
    """Load episodes CSV."""
    return pd.read_csv(csv_path)


def load_events(jsonl_path: str) -> List[dict]:
    """Load events JSONL."""
    events = []
    try:
        with open(jsonl_path, 'r') as f:
            for line in f:
                events.append(json.loads(line.strip()))
    except FileNotFoundError:
        pass
    return events


def compute_metrics(episodes_df: pd.DataFrame, events: List[dict]) -> Dict:
    """
    Compute evaluation metrics from episodes and events.
    
    Task-Aligned Metrics:
    - Success reflects actual task completion (cube placed, not just moved)
    - Penalties for collisions, penetration, repeated RECOVER
    - Separate tracking for different failure modes
    
    V11 Force Truth Metrics:
    - Proxy rate: % steps with force_signal_is_proxy=True
    - Impedance gate trigger count
    - Mean/max gated force value
    - Episodes with 100% proxy rate
    
    Returns:
        Dictionary of metrics
    """
    if len(episodes_df) == 0:
        return {
            'total_episodes': 0,
            'success_rate': 0.0,
            'collision_rate': 0.0,
            'self_collision_rate': 0.0,
            'table_collision_rate': 0.0,
            'mean_torque_sat_per_episode': 0.0,
            'mean_mpc_fail_per_episode': 0.0,
            'mean_mpc_solve_time_ms': 0.0,
            'max_mpc_solve_time_ms': 0.0,
            'mean_penetration': 0.0,
            'mean_episode_duration': 0.0,
            'mean_steps_per_episode': 0.0,
        }
    
    total = len(episodes_df)
    
    metrics = {
        'total_episodes': total,
        'success_rate': episodes_df['success'].sum() / total,
        'collision_rate': (episodes_df['collision_count'] > 0).sum() / total,
        'self_collision_rate': (episodes_df['self_collision_count'] > 0).sum() / total,
        'table_collision_rate': (episodes_df['table_collision_count'] > 0).sum() / total,
        'mean_torque_sat_per_episode': episodes_df['torque_sat_count'].mean(),
        'mean_mpc_fail_per_episode': episodes_df['mpc_fail_count'].mean(),
        'mean_mpc_solve_time_ms': episodes_df['mean_mpc_solve_ms'].mean(),
        'max_mpc_solve_time_ms': episodes_df['mean_mpc_solve_ms'].max(),
        'mean_penetration': episodes_df['max_penetration'].mean(),
        'mean_episode_duration': episodes_df['duration_s'].mean(),
        'mean_steps_per_episode': episodes_df['num_steps'].mean(),
    }
    
    # Task-aligned metrics: penalize safety violations
    # Episodes with collisions
    episodes_with_collisions = (episodes_df['collision_count'] > 0).sum()
    # Episodes with excessive penetration (>5 steps with penetration > 0.05m)
    excessive_penetration_episodes = 0
    if 'max_penetration' in episodes_df.columns:
        excessive_penetration_episodes = (episodes_df['max_penetration'] > 0.05).sum()
    
    metrics['episodes_with_safety_violations'] = episodes_with_collisions
    metrics['excessive_penetration_episodes'] = excessive_penetration_episodes
    
    # Failure reasons (categorized)
    failure_reasons = episodes_df[~episodes_df['success']]['failure_reason'].value_counts()
    metrics['failure_reasons'] = failure_reasons.to_dict() if len(failure_reasons) > 0 else {}
    
    # Count specific failure modes
    metrics['repeated_recover_failures'] = metrics['failure_reasons'].get('repeated_recover', 0)
    metrics['collision_failures'] = (
        metrics['failure_reasons'].get('self_collision', 0) + 
        metrics['failure_reasons'].get('table_collision', 0) +
        metrics['failure_reasons'].get('collision_during_task', 0) +
        metrics['failure_reasons'].get('excessive_collisions', 0)
    )
    
    # V6: Grasp validation metrics
    if 'grasp_attempts' in episodes_df.columns:
        metrics['total_grasp_attempts'] = episodes_df['grasp_attempts'].sum()
        metrics['total_grasp_confirmed'] = episodes_df['grasp_confirmed'].sum()
        metrics['total_false_lifts'] = episodes_df['false_lift_count'].sum()
        metrics['mean_grasp_confirmation_time_s'] = episodes_df[episodes_df['grasp_confirmation_time_s'] > 0]['grasp_confirmation_time_s'].mean() if (episodes_df['grasp_confirmation_time_s'] > 0).any() else 0.0
        metrics['total_slip_events'] = episodes_df['slip_events'].sum()
        
        # Calculate grasp success rate (confirmed / attempts)
        if metrics['total_grasp_attempts'] > 0:
            metrics['grasp_success_rate'] = metrics['total_grasp_confirmed'] / metrics['total_grasp_attempts']
        else:
            metrics['grasp_success_rate'] = 0.0
        
        # False lift rate (false lifts / confirmed grasps)
        if metrics['total_grasp_confirmed'] > 0:
            metrics['false_lift_rate'] = metrics['total_false_lifts'] / metrics['total_grasp_confirmed']
        else:
            metrics['false_lift_rate'] = 0.0
    
    # Event counts
    event_counts = {}
    for event in events:
        event_type = event['event_type']
        event_counts[event_type] = event_counts.get(event_type, 0) + 1
    metrics['event_counts'] = event_counts
    
    # V11: Force truth metrics from events
    impedance_gate_events = [e for e in events if e['event_type'] == 'impedance_force_gate_triggered']
    metrics['impedance_gate_trigger_count'] = len(impedance_gate_events)
    
    if impedance_gate_events:
        gate_values = []
        for e in impedance_gate_events:
            v = e.get('data', {}).get('gate_value', 0.0)
            try:
                v = float(v)
            except (TypeError, ValueError):
                continue
            if (v != v) or (v == float("inf")) or (v == float("-inf")):
                continue
            gate_values.append(v)

        metrics['mean_gate_force_value'] = sum(gate_values) / len(gate_values) if gate_values else 0.0
        metrics['max_gate_force_value'] = max(gate_values) if gate_values else 0.0
        
        # Count by source
        gate_sources = {}
        gate_proxy_count = 0
        for e in impedance_gate_events:
            source = e.get('data', {}).get('gate_source', 'unknown')
            gate_sources[source] = gate_sources.get(source, 0) + 1
            if e.get('data', {}).get('gate_proxy', False):
                gate_proxy_count += 1
        
        metrics['gate_sources'] = gate_sources
        metrics['gate_proxy_count'] = gate_proxy_count
    else:
        metrics['mean_gate_force_value'] = 0.0
        metrics['max_gate_force_value'] = 0.0
        metrics['gate_sources'] = {}
        metrics['gate_proxy_count'] = 0
    
    return metrics


def format_metrics(metrics: Dict) -> str:
    """Format metrics as a readable string with task-aligned focus."""
    lines = []
    lines.append("=" * 70)
    lines.append("EVALUATION METRICS (TASK-ALIGNED)")
    lines.append("=" * 70)
    lines.append(f"Total episodes:              {metrics['total_episodes']}")
    lines.append(f"Success rate:                {metrics['success_rate']:.1%}")
    lines.append("")
    
    lines.append("SAFETY VIOLATIONS (TASK PENALTIES):")
    lines.append(f"  Collision rate:            {metrics['collision_rate']:.1%}")
    lines.append(f"  Self-collision rate:       {metrics['self_collision_rate']:.1%}")
    lines.append(f"  Table-collision rate:      {metrics['table_collision_rate']:.1%}")
    lines.append(f"  Episodes with violations:  {metrics.get('episodes_with_safety_violations', 0)}")
    lines.append(f"  Excessive penetration:     {metrics.get('excessive_penetration_episodes', 0)}")
    lines.append("")
    
    lines.append("CONSTRAINTS:")
    lines.append(f"  Mean torque sat/episode:   {metrics['mean_torque_sat_per_episode']:.2f}")
    lines.append(f"  Mean MPC fails/episode:    {metrics['mean_mpc_fail_per_episode']:.2f}")
    lines.append(f"  Mean penetration:          {metrics['mean_penetration']:.4f} m")
    lines.append("")
    
    lines.append("MPC PERFORMANCE:")
    lines.append(f"  Mean solve time:           {metrics['mean_mpc_solve_time_ms']:.2f} ms")
    lines.append(f"  Max solve time:            {metrics['max_mpc_solve_time_ms']:.2f} ms")
    lines.append("")
    
    lines.append("EPISODE STATS:")
    lines.append(f"  Mean duration:             {metrics['mean_episode_duration']:.2f} s")
    lines.append(f"  Mean steps/episode:        {metrics['mean_steps_per_episode']:.1f}")
    lines.append("")
    
    # V6: Grasp validation metrics
    if 'total_grasp_attempts' in metrics:
        lines.append("GRASP VALIDATION METRICS (V6):")
        lines.append(f"  Total grasp attempts:      {metrics['total_grasp_attempts']}")
        lines.append(f"  Grasps confirmed:          {metrics['total_grasp_confirmed']}")
        lines.append(f"  Grasp success rate:        {metrics['grasp_success_rate']:.1%}")
        lines.append(f"  False lifts:               {metrics['total_false_lifts']}")
        lines.append(f"  False lift rate:           {metrics['false_lift_rate']:.1%}")
        lines.append(f"  Mean confirmation time:    {metrics['mean_grasp_confirmation_time_s']:.2f} s")
        lines.append(f"  Total slip events:         {metrics['total_slip_events']}")
        lines.append("")
    
    # V11: Force truth metrics
    if 'impedance_gate_trigger_count' in metrics:
        lines.append("FORCE TRUTH METRICS (V11):")
        lines.append(f"  Impedance gate triggers:   {metrics['impedance_gate_trigger_count']}")
        if metrics['impedance_gate_trigger_count'] > 0:
            lines.append(f"  Mean gated force:          {metrics['mean_gate_force_value']:.2f} N")
            lines.append(f"  Max gated force:           {metrics['max_gate_force_value']:.2f} N")
            lines.append(f"  Gate triggers from proxy:  {metrics['gate_proxy_count']}")
            if metrics.get('gate_sources'):
                lines.append(f"  Gate sources:")
                for source, count in metrics['gate_sources'].items():
                    lines.append(f"    {source}: {count}")
        lines.append("")
    
    if metrics.get('failure_reasons'):
        lines.append("FAILURE MODES (CATEGORIZED):")
        # Group by type
        collision_failures = metrics.get('collision_failures', 0)
        recover_failures = metrics.get('repeated_recover_failures', 0)
        if collision_failures > 0:
            lines.append(f"  Collision-related:         {collision_failures}")
        if recover_failures > 0:
            lines.append(f"  Repeated RECOVER loops:    {recover_failures}")
        
        # Show all failure reasons
        lines.append("  Detailed breakdown:")
        for reason, count in metrics['failure_reasons'].items():
            lines.append(f"    {reason}: {count}")
        lines.append("")
    
    if metrics.get('event_counts'):
        lines.append("EVENT COUNTS:")
        for event_type, count in metrics['event_counts'].items():
            lines.append(f"  {event_type}: {count}")
    
    lines.append("=" * 70)
    
    return "\n".join(lines)

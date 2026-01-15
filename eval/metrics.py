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
    
    # Failure reasons
    failure_reasons = episodes_df[~episodes_df['success']]['failure_reason'].value_counts()
    metrics['failure_reasons'] = failure_reasons.to_dict() if len(failure_reasons) > 0 else {}
    
    # Event counts
    event_counts = {}
    for event in events:
        event_type = event['event_type']
        event_counts[event_type] = event_counts.get(event_type, 0) + 1
    metrics['event_counts'] = event_counts
    
    return metrics


def format_metrics(metrics: Dict) -> str:
    """Format metrics as a readable string."""
    lines = []
    lines.append("=" * 70)
    lines.append("EVALUATION METRICS")
    lines.append("=" * 70)
    lines.append(f"Total episodes:              {metrics['total_episodes']}")
    lines.append(f"Success rate:                {metrics['success_rate']:.1%}")
    lines.append("")
    
    lines.append("COLLISIONS:")
    lines.append(f"  Collision rate:            {metrics['collision_rate']:.1%}")
    lines.append(f"  Self-collision rate:       {metrics['self_collision_rate']:.1%}")
    lines.append(f"  Table-collision rate:      {metrics['table_collision_rate']:.1%}")
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
    
    if metrics.get('failure_reasons'):
        lines.append("FAILURE REASONS:")
        for reason, count in metrics['failure_reasons'].items():
            lines.append(f"  {reason}: {count}")
        lines.append("")
    
    if metrics.get('event_counts'):
        lines.append("EVENT COUNTS:")
        for event_type, count in metrics['event_counts'].items():
            lines.append(f"  {event_type}: {count}")
    
    lines.append("=" * 70)
    
    return "\n".join(lines)

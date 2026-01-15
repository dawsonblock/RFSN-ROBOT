"""
Evaluation Report Generator
============================
Print summary statistics from runs.
"""

import sys
import os
from .metrics import load_episodes, load_events, compute_metrics, format_metrics


def main():
    """Generate report from a run directory."""
    if len(sys.argv) < 2:
        print("Usage: python -m eval.report <run_dir>")
        print("Example: python -m eval.report runs/20260115_001234")
        sys.exit(1)
    
    run_dir = sys.argv[1]
    
    if not os.path.exists(run_dir):
        print(f"Error: Run directory not found: {run_dir}")
        sys.exit(1)
    
    episodes_path = os.path.join(run_dir, "episodes.csv")
    events_path = os.path.join(run_dir, "events.jsonl")
    
    if not os.path.exists(episodes_path):
        print(f"Error: episodes.csv not found in {run_dir}")
        sys.exit(1)
    
    # Load data
    print(f"Loading data from: {run_dir}")
    episodes_df = load_episodes(episodes_path)
    events = load_events(events_path)
    
    print(f"Loaded {len(episodes_df)} episodes, {len(events)} events")
    print()
    
    # Compute metrics
    metrics = compute_metrics(episodes_df, events)
    
    # Print formatted report
    print(format_metrics(metrics))


if __name__ == "__main__":
    main()

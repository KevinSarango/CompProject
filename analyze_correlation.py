#!/usr/bin/env python3
"""
analyze_correlation.py

Correlate RL learning performance with network traffic patterns.

Reads:
  - rl_episode_rewards.log: RL agent episode rewards and learning metrics
  - rl_metrics_log.csv: Controller-side flow statistics per interval

  These files can be located either in the project root or under the data/ directory.

Outputs:
  - Console analysis report
  - correlation_report.txt: Detailed findings
  - Visualizations (if matplotlib available)

Usage:
    python3 analyze_correlation.py
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

LOG_PATHS = {
    'rewards': [
        'rl_episode_rewards.log',
        os.path.join('data', 'rl_episode_rewards.log')
    ],
    'metrics': [
        'rl_metrics_log.csv',
        os.path.join('data', 'rl_metrics_log.csv')
    ]
}


def _find_log_file(candidates):
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def load_logs():
    """Load all available logs."""
    data = {}
    
    rewards_path = _find_log_file(LOG_PATHS['rewards'])
    if rewards_path is not None:
        data['rewards'] = pd.read_csv(rewards_path)
        print(f"✓ Loaded {rewards_path} ({len(data['rewards'])} episodes)")
    else:
        print("✗ Missing rl_episode_rewards.log")
        data['rewards'] = None
    
    metrics_path = _find_log_file(LOG_PATHS['metrics'])
    if metrics_path is not None:
        data['metrics'] = pd.read_csv(metrics_path, names=[
            'timestamp', 'num_switches', 'num_flows', 'flow_stats_update_ms',
            'total_duration', 'total_idle_time', 'flow_count', 'total_packet_count', 'total_byte_count'
        ])
        print(f"✓ Loaded {metrics_path} ({len(data['metrics'])} samples)")
    else:
        print("✗ Missing rl_metrics_log.csv")
        data['metrics'] = None
    
    return data


def analyze_rl_learning(rewards_df):
    """Analyze RL learning curve."""
    if rewards_df is None:
        return
    
    print("\n" + "=" * 70)
    print("  RL LEARNING ANALYSIS")
    print("=" * 70)
    
    if 'reward' not in rewards_df.columns:
        print("RL rewards file missing 'reward' column.")
        print(f"Available columns: {', '.join(rewards_df.columns)}")
        return

    print(f"\nTraining statistics:")
    print(f"  Total episodes:    {len(rewards_df)}")
    print(f"  Initial reward:    {rewards_df['reward'].iloc[0]:+.4f}")
    print(f"  Final reward:      {rewards_df['reward'].iloc[-1]:+.4f}")
    print(f"  Mean reward:       {rewards_df['reward'].mean():+.4f}")
    print(f"  Std reward:        {rewards_df['reward'].std():.4f}")
    print(f"  Min reward:        {rewards_df['reward'].min():+.4f}")
    print(f"  Max reward:        {rewards_df['reward'].max():+.4f}")
    
    early = rewards_df['reward'].iloc[:max(5, len(rewards_df)//4)].mean()
    late = rewards_df['reward'].iloc[-max(5, len(rewards_df)//4):].mean()
    trend = late - early
    
    print(f"\n  Learning trend:")
    print(f"    Early episodes avg: {early:+.4f}")
    print(f"    Late episodes avg:  {late:+.4f}")
    print(f"    Improvement:        {trend:+.4f}")
    
    if trend > 0.1:
        print(f"    → ✓ STRONG POSITIVE LEARNING")
    elif trend > 0.02:
        print(f"    → ✓ Positive learning")
    elif trend > -0.02:
        print(f"    → ≈ Neutral (oscillating)")
    elif trend > -0.1:
        print(f"    → ✗ Slight decay")
    else:
        print(f"    → ✗✗ STRONG DECAY - CHECK TRAFFIC CONDITIONS")
    
    if 'decay' in rewards_df.columns:
        improving_episodes = rewards_df[rewards_df['decay'] > 0.1]
        if len(improving_episodes) > 0:
            episodes = improving_episodes['episode'].tolist()[:10] if 'episode' in improving_episodes.columns else []
            print(f"\n  Learning improvement detected in {len(improving_episodes)} episodes:")
            print(f"    Episodes: {episodes}")
    else:
        print("\n  No decay column present in RL rewards log; skipping improvement analysis.")

def analyze_controller_metrics(metrics_df):
    """Analyze controller-side flow dynamics."""
    if metrics_df is None:
        return
    
    print("\n" + "=" * 70)
    print("  CONTROLLER METRICS ANALYSIS")
    print("=" * 70)
    
    print(f"\nFlow table statistics:")
    print(f"  Avg concurrent flows:    {metrics_df['num_flows'].mean():.1f}")
    print(f"  Max concurrent flows:    {metrics_df['num_flows'].max():.0f}")
    print(f"  Min concurrent flows:    {metrics_df['num_flows'].min():.0f}")
    print(f"  Std dev:                 {metrics_df['num_flows'].std():.1f}")
    
    print(f"\nController overhead:")
    print(f"  Avg stats update time:   {metrics_df['flow_stats_update_ms'].mean():.2f} ms")
    print(f"  Max stats update time:   {metrics_df['flow_stats_update_ms'].max():.2f} ms")
    
    print(f"\nFlow statistics:")
    print(f"  Total flow count:        {metrics_df['flow_count'].sum():.0f}")
    print(f"  Avg total packets:       {metrics_df['total_packet_count'].mean():.0f}")
    print(f"  Avg total bytes:         {metrics_df['total_byte_count'].mean():.0f}")
    print(f"  Avg total duration:      {metrics_df['total_duration'].mean():.2f} s")
    print(f"  Avg total idle time:     {metrics_df['total_idle_time'].mean():.2f} s")
    
    # Flow dynamics during learning
    if 'num_flows' in metrics_df.columns:
        high_flow = metrics_df[metrics_df['num_flows'] > metrics_df['num_flows'].quantile(0.75)]
        print(f"\n  High flow episodes: {len(high_flow)} (top 25%)")
        print(f"    Avg flows during: {high_flow['num_flows'].mean():.1f}")
        if high_flow['num_flows'].mean() > 50:
            print(f"    ⚠ Very high flow count - may impact RL learning speed")
    else:
        print("\n  No num_flows column present in controller metrics log.")

def correlate_metrics(rewards_df, metrics_df):
    """Cross-correlate metrics."""
    if rewards_df is None or metrics_df is None:
        return
    
    print("\n" + "=" * 70)
    print("  CORRELATION ANALYSIS")
    print("=" * 70)
    
    avg_reward = rewards_df['reward'].mean()
    
    print(f"\nReward vs controller metrics:")
    print(f"  Average episode reward: {avg_reward:+.4f}")
    print(f"  Total metric samples: {len(metrics_df)}")
    
    if 'num_flows' in metrics_df.columns:
        print(f"  Avg concurrent flows: {metrics_df['num_flows'].mean():.1f}")
        if metrics_df['num_flows'].mean() > 50:
            print(f"  → High flow volume may impact RL learning speed")
    
    print(f"\nState space exploration:")
    if 'num_states' in rewards_df.columns:
        print(f"  Unique states explored: {rewards_df['num_states'].max():.0f}")
    else:
        print("  Unique states explored: unavailable")
    if 'epsilon' in rewards_df.columns:
        print(f"  Final exploration rate: {rewards_df['epsilon'].iloc[-1]:.4f}")
    else:
        print("  Final exploration rate: unavailable")

    if 'num_states' in rewards_df.columns:
        if rewards_df['num_states'].max() > 100:
            print(f"  → Rich state space - agent has learned diverse policies")
        else:
            print(f"  → Limited state space - may indicate narrow traffic patterns")
    else:
        print(f"  → No num_states data to evaluate exploration depth.")

def generate_report(data):
    """Generate detailed correlation report."""
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("  RL TRAINING CORRELATION REPORT")
    report_lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 70)
    report_lines.append("")
    
    # RL learning results
    if data['rewards'] is not None:
        rewards_df = data['rewards']
        report_lines.append("RL LEARNING RESULTS")
        report_lines.append("-" * 70)
        report_lines.append(f"Episodes trained: {len(rewards_df)}")
        report_lines.append(f"Reward range: [{rewards_df['reward'].min():.4f}, {rewards_df['reward'].max():.4f}]")
        report_lines.append(f"Mean reward: {rewards_df['reward'].mean():.4f}")
        
        early = rewards_df['reward'].iloc[:max(5, len(rewards_df)//4)].mean()
        late = rewards_df['reward'].iloc[-max(5, len(rewards_df)//4):].mean()
        improvement = late - early
        report_lines.append(f"Learning improvement: {improvement:+.4f}")
        report_lines.append(f"Final epsilon: {rewards_df['epsilon'].iloc[-1]:.5f}")
        report_lines.append(f"States discovered: {rewards_df['num_states'].max():.0f}")
        report_lines.append("")
    
    # Recommendations
    report_lines.append("RECOMMENDATIONS")
    report_lines.append("-" * 70)
    
    if data['rewards'] is not None:
        if improvement > 0:
            report_lines.append("• Learning is improving: continue training with the current setup.")
        elif improvement > -0.02:
            report_lines.append("• Learning is stable: inspect queue policy actions for oscillations.")
        else:
            report_lines.append("• Learning is degrading: review reward shaping and flow prioritization.")
    else:
        report_lines.append("• No reward data available for recommendations.")
    
    report_lines.append("")
    
    report_text = "\n".join(report_lines)
    return report_text

def main():
    print("\n" + "=" * 70)
    print("  RL TRAINING CORRELATION ANALYZER")
    print("=" * 70)
    print("\nLoading logs...\n")
    
    data = load_logs()
    
    if data['rewards'] is None and data['metrics'] is None:
        print("\n✗ No log files found. Run your network simulation first.")
        print("  Expected files: rl_episode_rewards.log, rl_metrics_log.csv")
        sys.exit(1)
    
    # Run analyses
    analyze_rl_learning(data['rewards'])
    analyze_controller_metrics(data['metrics'])
    correlate_metrics(data['rewards'], data['metrics'])
    
    # Generate report file
    report = generate_report(data)
    with open('correlation_report.txt', 'w') as f:
        f.write(report)
    
    print("\n" + "=" * 70)
    print(f"✓ Report saved to correlation_report.txt")
    print("=" * 70 + "\n")
    
    # Try to generate plots if matplotlib available
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        if data['rewards'] is not None:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle('RL Training Correlation Analysis')
            rewards_df = data['rewards']

            # Plot 1: Reward trend
            ax = axes[0, 0]
            ax.plot(rewards_df['episode'], rewards_df['reward'], 'b-', alpha=0.5, label='Episode reward')
            if 'avg_10' in rewards_df.columns:
                ax.plot(rewards_df['episode'], rewards_df['avg_10'], 'r-', linewidth=2, label='Avg 10 episodes')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Reward')
            ax.set_title('Learning Curve')
            ax.legend()
            ax.grid()
            
            # Plot 2: Decay detection
            ax = axes[0, 1]
            if 'decay' in rewards_df.columns:
                ax.plot(rewards_df['episode'], rewards_df['decay'], 'g-', label='Learning improvement')
                ax.axhline(y=0.1, color='r', linestyle='--', label='Improvement threshold')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Improvement (avg20 - avg10)')
            ax.set_title('Learning Progress')
            ax.legend()
            ax.grid()
            
            # Plot 3: State exploration
            ax = axes[1, 0]
            if 'num_states' in rewards_df.columns:
                ax.plot(rewards_df['episode'], rewards_df['num_states'], 'purple', label='Q-table states')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Number of States')
            ax.set_title('State Space Exploration')
            ax.legend()
            ax.grid()
            
            # Plot 4: Exploration decay
            ax = axes[1, 1]
            if 'epsilon' in rewards_df.columns:
                ax.plot(rewards_df['episode'], rewards_df['epsilon'], 'orange', label='Epsilon (exploration)')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Epsilon')
            ax.set_title('Exploration Rate Decay')
            ax.legend()
            ax.grid()
            
            plt.tight_layout()
            plt.savefig('rl_training_analysis.png', dpi=100)
            print("✓ Plot saved to rl_training_analysis.png\n")
    except ImportError:
        print("\n(matplotlib not available - skipping plots)")
        print("  Install: pip install matplotlib\n")

if __name__ == '__main__':
    main()

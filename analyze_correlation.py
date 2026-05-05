#!/usr/bin/env python3
"""
analyze_correlation.py

Correlate RL learning performance with network traffic patterns.

Reads:
  - traffic_scenario.log: What flows were sent and their characteristics
  - rl_episode_rewards.log: RL agent episode rewards and learning metrics
  - rl_metrics_log.csv: Controller-side flow statistics per interval

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

def load_logs():
    """Load all available logs."""
    data = {}
    
    # Load traffic scenario
    if os.path.exists('traffic_scenario.log'):
        data['traffic'] = pd.read_csv('traffic_scenario.log')
        print(f"✓ Loaded traffic_scenario.log ({len(data['traffic'])} flows)")
    else:
        print("✗ Missing traffic_scenario.log")
        data['traffic'] = None
    
    # Load RL episode rewards
    if os.path.exists('rl_episode_rewards.log'):
        data['rewards'] = pd.read_csv('rl_episode_rewards.log')
        print(f"✓ Loaded rl_episode_rewards.log ({len(data['rewards'])} episodes)")
    else:
        print("✗ Missing rl_episode_rewards.log")
        data['rewards'] = None
    
    # Load controller metrics
    if os.path.exists('rl_metrics_log.csv'):
        data['metrics'] = pd.read_csv('rl_metrics_log.csv')
        print(f"✓ Loaded rl_metrics_log.csv ({len(data['metrics'])} samples)")
    else:
        print("✗ Missing rl_metrics_log.csv")
        data['metrics'] = None
    
    return data

def analyze_traffic(traffic_df):
    """Analyze traffic scenario."""
    if traffic_df is None:
        return
    
    print("\n" + "=" * 70)
    print("  TRAFFIC SCENARIO ANALYSIS")
    print("=" * 70)
    
    traffic_counts = traffic_df['flow_type'].value_counts()
    print(f"\nFlow types sent:")
    for ftype, count in traffic_counts.items():
        subset = traffic_df[traffic_df['flow_type'] == ftype]
        bitrates = subset['bitrate_mbps'].values
        print(f"  {ftype:15s}: {count:2d} flows @ {bitrates[0]:6.3f} Mbps")
    
    print(f"\nTotal flows: {len(traffic_df)}")
    print(f"Bottleneck: 10 Mbps")
    total_bitrate = traffic_df['bitrate_mbps'].sum()
    print(f"Total traffic demand: {total_bitrate:.2f} Mbps")
    if total_bitrate > 10:
        print(f"  ⚠ Exceeds bottleneck by {total_bitrate - 10:.2f} Mbps")
        print(f"    → Expect packet loss and queuing")
    print()

def analyze_rl_learning(rewards_df):
    """Analyze RL learning curve."""
    if rewards_df is None:
        return
    
    print("\n" + "=" * 70)
    print("  RL LEARNING ANALYSIS")
    print("=" * 70)
    
    print(f"\nTraining statistics:")
    print(f"  Total episodes:    {len(rewards_df)}")
    print(f"  Initial reward:    {rewards_df['reward'].iloc[0]:+.4f}")
    print(f"  Final reward:      {rewards_df['reward'].iloc[-1]:+.4f}")
    print(f"  Mean reward:       {rewards_df['reward'].mean():+.4f}")
    print(f"  Std reward:        {rewards_df['reward'].std():.4f}")
    print(f"  Min reward:        {rewards_df['reward'].min():+.4f}")
    print(f"  Max reward:        {rewards_df['reward'].max():+.4f}")
    
    # Learning trend
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
    
    # Decay episodes
    decay_episodes = rewards_df[rewards_df['decay'] < -0.1]
    if len(decay_episodes) > 0:
        print(f"\n  Learning decay detected in {len(decay_episodes)} episodes:")
        print(f"    Episodes: {decay_episodes['episode'].tolist()[:10]}")
        print(f"    → Compare with traffic_scenario.log timestamps")

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
    
    # Flow dynamics during learning
    if 'num_flows' in metrics_df.columns:
        high_flow = metrics_df[metrics_df['num_flows'] > metrics_df['num_flows'].quantile(0.75)]
        print(f"\n  High flow episodes: {len(high_flow)} (top 25%)")
        print(f"    Avg flows during: {high_flow['num_flows'].mean():.1f}")
        if high_flow['num_flows'].mean() > 50:
            print(f"    ⚠ Very high flow count - may impact RL learning speed")

def correlate_metrics(traffic_df, rewards_df, metrics_df):
    """Cross-correlate metrics."""
    if traffic_df is None or rewards_df is None:
        return
    
    print("\n" + "=" * 70)
    print("  CORRELATION ANALYSIS")
    print("=" * 70)
    
    # Traffic load vs learning
    total_bitrate = traffic_df['bitrate_mbps'].sum()
    avg_reward = rewards_df['reward'].mean()
    
    print(f"\nTraffic load impact on learning:")
    print(f"  Total traffic demand: {total_bitrate:.2f} Mbps")
    print(f"  Bottleneck capacity: 10 Mbps")
    congestion_ratio = min(total_bitrate / 10, 1.0)
    print(f"  Congestion ratio:     {congestion_ratio:.1%}")
    print(f"  Average episode reward: {avg_reward:+.4f}")
    
    if congestion_ratio > 0.8:
        print(f"\n  → High congestion: RL may struggle to learn optimal policies")
        print(f"    Suggest: Reduce traffic or increase bottleneck capacity")
    else:
        print(f"\n  → Moderate congestion: Ideal for RL training")
    
    # State space exploration
    print(f"\nState space exploration:")
    print(f"  Unique states explored: {rewards_df['num_states'].max():.0f}")
    print(f"  Final exploration rate: {rewards_df['epsilon'].iloc[-1]:.4f}")
    
    if rewards_df['num_states'].max() > 100:
        print(f"  → Rich state space - agent has learned diverse policies")
    else:
        print(f"  → Limited state space - may indicate narrow traffic patterns")

def generate_report(data):
    """Generate detailed correlation report."""
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("  RL TRAINING CORRELATION REPORT")
    report_lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 70)
    report_lines.append("")
    
    # Traffic scenario
    if data['traffic'] is not None:
        traffic_df = data['traffic']
        report_lines.append("TRAFFIC SCENARIO")
        report_lines.append("-" * 70)
        report_lines.append(f"Total flows: {len(traffic_df)}")
        for ftype in traffic_df['flow_type'].unique():
            count = len(traffic_df[traffic_df['flow_type'] == ftype])
            bitrate = traffic_df[traffic_df['flow_type'] == ftype]['bitrate_mbps'].iloc[0]
            report_lines.append(f"  {ftype}: {count} flow(s) @ {bitrate} Mbps")
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
    
    if data['traffic'] is not None and data['rewards'] is not None:
        total_bitrate = data['traffic']['bitrate_mbps'].sum()
        improvement = late - early if 'late' in locals() else 0
        
        if total_bitrate > 15 and improvement < 0:
            report_lines.append("• High traffic + negative learning:")
            report_lines.append("  → Reduce number of concurrent flows")
            report_lines.append("  → Increase bottleneck link bandwidth")
            report_lines.append("  → Train for more episodes with lower learning rate")
        elif total_bitrate > 10:
            report_lines.append("• High traffic congestion detected:")
            report_lines.append("  → Consider this a challenging scenario")
            report_lines.append("  → Monitor how RL prioritizes different flow types")
        else:
            report_lines.append("• Traffic load is reasonable for RL training")
            report_lines.append("  → Focus on analyzing policy decisions")
    
    report_lines.append("")
    
    report_text = "\n".join(report_lines)
    return report_text

def main():
    print("\n" + "=" * 70)
    print("  RL TRAINING CORRELATION ANALYZER")
    print("=" * 70)
    print("\nLoading logs...\n")
    
    data = load_logs()
    
    if data['traffic'] is None and data['rewards'] is None and data['metrics'] is None:
        print("\n✗ No log files found. Run your network simulation first.")
        print("  Expected files: traffic_scenario.log, rl_episode_rewards.log, rl_metrics_log.csv")
        sys.exit(1)
    
    # Run analyses
    analyze_traffic(data['traffic'])
    analyze_rl_learning(data['rewards'])
    analyze_controller_metrics(data['metrics'])
    correlate_metrics(data['traffic'], data['rewards'], data['metrics'])
    
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
            
            # Plot 1: Reward trend
            ax = axes[0, 0]
            rewards_df = data['rewards']
            ax.plot(rewards_df['episode'], rewards_df['reward'], 'b-', alpha=0.5, label='Episode reward')
            ax.plot(rewards_df['episode'], rewards_df['avg_10'], 'r-', linewidth=2, label='Avg 10 episodes')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Reward')
            ax.set_title('Learning Curve')
            ax.legend()
            ax.grid()
            
            # Plot 2: Decay detection
            ax = axes[0, 1]
            ax.plot(rewards_df['episode'], rewards_df['decay'], 'g-', label='Learning decay')
            ax.axhline(y=-0.1, color='r', linestyle='--', label='Decay threshold')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Decay (avg10 - avg20)')
            ax.set_title('Learning Stability')
            ax.legend()
            ax.grid()
            
            # Plot 3: State exploration
            ax = axes[1, 0]
            ax.plot(rewards_df['episode'], rewards_df['num_states'], 'purple', label='Q-table states')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Number of States')
            ax.set_title('State Space Exploration')
            ax.legend()
            ax.grid()
            
            # Plot 4: Exploration decay
            ax = axes[1, 1]
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

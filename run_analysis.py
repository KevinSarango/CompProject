#!/usr/bin/env python3
"""
QUICK START: Run this after training completes to see correlation analysis
"""

import subprocess
import sys

print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                 RL TRAINING CORRELATION ANALYSIS - QUICK START              ║
╚══════════════════════════════════════════════════════════════════════════════╝

This script provides a guided analysis of your RL training results.

Generated Files:
  ✓ traffic_scenario.log         - What flows were sent
  ✓ rl_episode_rewards.log       - How RL learned over time  
  ✓ rl_metrics_log.csv           - Controller overhead metrics

Next Steps:
""")

print("\n1. ANALYZE CORRELATION")
print("   " + "─" * 72)
print("   python3 analyze_correlation.py")
print("   → Generates: correlation_report.txt, rl_training_analysis.png")

print("\n2. READ THE DETAILED REPORT")
print("   " + "─" * 72)
print("   cat correlation_report.txt")
print("   → Shows: traffic load impact on learning, recommendations")

print("\n3. CHECK FOR LEARNING DECAY")
print("   " + "─" * 72)
print("   grep 'DECAY' rl_episode_rewards.log")
print("   → Episodes where learning degraded (avg_10 - avg_20 < -0.1)")

print("\n4. COMPARE TRAFFIC VS LEARNING")
print("   " + "─" * 72)
print("   head traffic_scenario.log")
print("   tail rl_episode_rewards.log")
print("   → See what flows caused what learning results")

print("\n5. VIEW VISUALIZATION (if available)")
print("   " + "─" * 72)
print("   open rl_training_analysis.png")
print("   → 4-panel chart of learning dynamics")

print("\n" + "═" * 76)
print("\nKEY QUESTIONS TO ANSWER:")
print("  Q1: Did the agent learn? → Check avg_10 trend in rl_episode_rewards.log")
print("  Q2: When did learning degrade? → Search for DECAY flag")
print("  Q3: What traffic caused decay? → Cross-check with traffic_scenario.log")
print("  Q4: Is congestion the problem? → Check total_bitrate vs 10 Mbps bottleneck")
print("  Q5: Is controller overhead high? → Check num_flows and stats_update_ms")

print("\n" + "═" * 76)
print("RUN NOW: python3 analyze_correlation.py")
print("═" * 76 + "\n")

# Auto-run if matplotlib is available
try:
    import pandas as pd
    subprocess.run(['python3', 'analyze_correlation.py'], check=True)
except Exception as e:
    print(f"\nNote: Could not auto-run: {e}")
    print("Run manually: python3 analyze_correlation.py\n")

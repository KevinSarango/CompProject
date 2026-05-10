#!/bin/bash

PROJECT_DIR="$HOME/CompProject"

echo "========================================="
echo " Multipath SDN RL Project Instructions"
echo "========================================="
echo

echo "Step 1: Generate TrafPy-style traffic demands"
echo "cd $PROJECT_DIR"
echo "python3 generate_trafpy_demands.py"
echo

echo "Step 2: Train RL agent on the same traffic demands"
echo "python3 train_rl_agent.py"
echo "python3 plot_training_results.py"
echo

echo "Step 3: Run FIFO baseline"
echo
echo "Terminal 1:"
echo "cd $PROJECT_DIR"
echo "sudo mn -c"
echo "source ~/ryu38/bin/activate"
echo "ryu-manager --verbose ryu_fifo_controller.py"
echo
echo "Terminal 2:"
echo "cd $PROJECT_DIR"
echo "sudo python3 diamond_topology.py --policy FIFO"
echo
echo "When tests finish:"
echo "Inside Mininet, type: exit"
echo "Then stop Ryu with CTRL+C"
echo

echo "Step 4: Run RL controller"
echo
echo "Terminal 1:"
echo "cd $PROJECT_DIR"
echo "sudo mn -c"
echo "source ~/ryu38/bin/activate"
echo "ryu-manager --verbose ryu_rl_controller.py"
echo
echo "Terminal 2:"
echo "cd $PROJECT_DIR"
echo "sudo python3 diamond_topology.py --policy RL"
echo
echo "When tests finish:"
echo "Inside Mininet, type: exit"
echo "Then stop Ryu with CTRL+C"
echo

echo "Step 5: Evaluate FIFO vs RL"
echo "python3 eval_fifo_vs_rl.py"
echo

echo "Important output files:"
echo "- data/trafpy_demands.csv"
echo "- data/q_table.json"
echo "- data/training_rewards.csv"
echo "- data/training_steps.csv"
echo "- data/fifo_metrics.csv"
echo "- data/rl_metrics.csv"
echo "- data/fifo_traffic_metrics.csv"
echo "- data/rl_traffic_metrics.csv"
echo "- data/plots/"
echo

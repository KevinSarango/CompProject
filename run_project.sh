#!/bin/bash

PROJECT_DIR="$HOME/CompProject"
RYU_ENV="$HOME/ryu38"

echo "========================================="
echo "          Project Instructions"
echo "========================================="
echo

echo "========================================="
echo " Step 1: Train RL model"
echo "========================================="
echo "cd $PROJECT_DIR"
echo "python3 train_rl_agent.py"
echo
echo "This creates:"
echo "- data/q_table.json"
echo "- data/training_rewards.csv"
echo "- data/training_steps.csv"
echo

echo "========================================="
echo " Step 2: Generate training graphs"
echo "========================================="
echo "cd $PROJECT_DIR"
echo "python3 plot_training_results.py"
echo
echo "This creates:"
echo "- data/plots/reward_curve_total.png"
echo "- data/plots/reward_curve_average.png"
echo "- data/plots/q_table_values.png"
echo

echo "========================================="
echo " Step 3: FIFO baseline"
echo "========================================="
echo "Terminal 1:"
echo "cd $PROJECT_DIR"
echo "sudo mn -c"
echo "source $RYU_ENV/bin/activate"
echo "ryu-manager --verbose ryu_fifo_controller.py"
echo
echo "Terminal 2:"
echo "cd $PROJECT_DIR"
echo "sudo python3 diamond_topology.py --policy FIFO"
echo
echo "FIFO decisions saved to data/fifo_metrics.csv"
echo

echo "========================================="
echo " Step 4: RL controller"
echo "========================================="
echo "Stop Mininet with: exit"
echo "Stop Ryu with: CTRL+C"
echo
echo "Terminal 1:"
echo "cd $PROJECT_DIR"
echo "sudo mn -c"
echo "source $RYU_ENV/bin/activate"
echo "ryu-manager --verbose ryu_rl_controller.py"
echo
echo "Terminal 2:"
echo "cd $PROJECT_DIR"
echo "sudo python3 diamond_topology.py --policy RL"
echo
echo "RL decisions saved to data/rl_metrics.csv"
echo

echo "========================================="
echo " Step 5: Evaluate FIFO vs RL"
echo "========================================="
echo "cd $PROJECT_DIR"
echo "python3 eval_fifo_vs_rl.py"
echo

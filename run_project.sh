#!/bin/bash

PROJECT_DIR="$HOME/CompProject"

echo "========================================="
echo " Multipath SDN RL Project Instructions"
echo "========================================="
echo

echo "Step 0: Go to the project directory"
echo "cd $PROJECT_DIR"
echo

echo "Step 1: Train RL with reproducible per-episode traffic"
echo "The training script now generates a different traffic demand trace for every episode."
echo "It uses 1000 episodes by default and saves the per-episode seed manifest so retraining is reproducible."
echo "You do NOT need to run generate_trafpy_demands.py before training."
echo
echo "For diamond:"
echo "TOPO_MODE=diamond python3 train_rl_agent.py"
echo "TOPO_MODE=diamond python3 plot_training_results.py"
echo "Seed manifest: data/training_episode_seeds_diamond.csv"
echo
echo "For three-path:"
echo "TOPO_MODE=three_path python3 train_rl_agent.py"
echo "TOPO_MODE=three_path python3 plot_training_results.py"
echo "Seed manifest: data/training_episode_seeds_three_path.csv"
echo
echo "Optional: generate one standalone demand file manually only if you want to inspect or run custom traffic outside training:"
echo "python3 generate_trafpy_demands.py"
echo

echo "========================================="
echo " DIAMOND TOPOLOGY"
echo "========================================="
echo

echo "FIFO diamond:"
echo "Terminal 1:"
echo "cd $PROJECT_DIR"
echo "sudo mn -c"
echo "source ~/ryu38/bin/activate"
echo "TOPO_MODE=diamond ryu-manager --verbose ryu_fifo_controller.py"
echo
echo "Terminal 2:"
echo "cd $PROJECT_DIR"
echo "sudo python3 diamond_topology.py --policy FIFO"
echo

echo "RL diamond:"
echo "Terminal 1:"
echo "cd $PROJECT_DIR"
echo "sudo mn -c"
echo "source ~/ryu38/bin/activate"
echo "TOPO_MODE=diamond ryu-manager --verbose ryu_rl_controller.py"
echo
echo "Terminal 2:"
echo "cd $PROJECT_DIR"
echo "sudo python3 diamond_topology.py --policy RL"
echo

echo "Evaluate diamond:"
echo "TOPO_MODE=diamond python3 eval_fifo_vs_rl.py"
echo

echo "========================================="
echo " THREE-PATH TOPOLOGY"
echo "========================================="
echo

echo "FIFO three-path:"
echo "Terminal 1:"
echo "cd $PROJECT_DIR"
echo "sudo mn -c"
echo "source ~/ryu38/bin/activate"
echo "TOPO_MODE=three_path ryu-manager --verbose ryu_fifo_controller.py"
echo
echo "Terminal 2:"
echo "cd $PROJECT_DIR"
echo "sudo python3 three_path_topology.py --policy FIFO"
echo

echo "RL three-path:"
echo "Terminal 1:"
echo "cd $PROJECT_DIR"
echo "sudo mn -c"
echo "source ~/ryu38/bin/activate"
echo "TOPO_MODE=three_path ryu-manager --verbose ryu_rl_controller.py"
echo
echo "Terminal 2:"
echo "cd $PROJECT_DIR"
echo "sudo python3 three_path_topology.py --policy RL"
echo

echo "Evaluate three-path:"
echo "TOPO_MODE=three_path python3 eval_fifo_vs_rl.py"
echo

echo "Important output files:"
echo "- data/q_table_diamond.json"
echo "- data/q_table_three_path.json"
echo "- data/training_episode_seeds_diamond.csv"
echo "- data/training_episode_seeds_three_path.csv"
echo "- data/training_rewards_diamond.csv"
echo "- data/training_rewards_three_path.csv"
echo "- data/diamond_fifo_metrics.csv"
echo "- data/diamond_rl_metrics.csv"
echo "- data/three_path_fifo_metrics.csv"
echo "- data/three_path_rl_metrics.csv"
echo "- data/plots/"
echo

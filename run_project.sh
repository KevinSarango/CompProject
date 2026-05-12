#!/bin/bash

PROJECT_DIR="$HOME/CompProject"

cat <<'EOF'
=========================================
 Multipath SDN RL Project Instructions
=========================================

The RL trainer now generates fresh, reproducible traffic demands per training
episode. You do NOT need to run generate_trafpy_demands.py before training.

The Mininet automated evaluation now runs multiple deterministic test seeds per
policy. Control this with:
  EVAL_NUM_RUNS   number of evaluation traces, default 5
  EVAL_NUM_FLOWS  flows per evaluation trace, default 150
  EVAL_BASE_SEED  first deterministic test seed, default 9000

The learned-policy plots now show only visited Q-table states by default.
Control this with:
  MIN_STATE_VISITS_FOR_POLICY_PLOT default 1
  MAX_POLICY_STATES_TO_PLOT        default 300

EOF

echo

echo "========================================="
echo " DIAMOND TOPOLOGY"
echo "========================================="
echo

echo "Train RL for diamond:"
echo "cd $PROJECT_DIR"
echo "TOPO_MODE=diamond python3 train_rl_agent.py"
echo "TOPO_MODE=diamond python3 plot_training_results.py"
echo

echo "FIFO diamond multi-seed evaluation:"
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

echo "RL diamond multi-seed evaluation:"
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

echo "Train RL for three-path:"
echo "cd $PROJECT_DIR"
echo "TOPO_MODE=three_path python3 train_rl_agent.py"
echo "TOPO_MODE=three_path python3 plot_training_results.py"
echo

echo "FIFO three-path multi-seed evaluation:"
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

echo "RL three-path multi-seed evaluation:"
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

echo "Useful output files:"
echo "- data/q_table_diamond.json"
echo "- data/q_table_three_path.json"
echo "- data/state_visit_counts_diamond.json"
echo "- data/state_visit_counts_three_path.json"
echo "- data/diamond_fifo_traffic_metrics.csv"
echo "- data/diamond_rl_traffic_metrics.csv"
echo "- data/three_path_fifo_traffic_metrics.csv"
echo "- data/three_path_rl_traffic_metrics.csv"
echo "- data/diamond_fifo_vs_rl_summary.csv"
echo "- data/three_path_fifo_vs_rl_summary.csv"
echo "- data/plots/"
echo

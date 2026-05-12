#!/bin/bash

PROJECT_DIR="$HOME/CompProject"

cat <<'EOF'
=========================================
 Multipath SDN RL Project Instructions
=========================================

This branch keeps the Sebas state shape:
  least_utilized_path_demand_bin_previous_action

New changes in this patch:
  1. The RL Ryu controller reads real flow sizes from data/trafpy_demands.csv.
     It maps TCP port 5001 + flow_id back to size_kb.
  2. Automated Mininet evaluation runs multiple deterministic test seeds.
  3. eval_fifo_vs_rl.py writes summary metrics and summary plots.
  4. plot_training_results.py plots learned policy/Q-values for visited states.

Evaluation controls:
  EVAL_NUM_RUNS   number of evaluation traces, default 5
  EVAL_NUM_FLOWS  flows per evaluation trace, default 150
  EVAL_BASE_SEED  first deterministic test seed, default 9000

Policy plot controls:
  MIN_STATE_VISITS_FOR_POLICY_PLOT  default 1
  MAX_POLICY_STATES_TO_PLOT         default 200

=========================================
 Before training
=========================================
cd ~/CompProject
python3 generate_trafpy_demands.py

=========================================
 DIAMOND TOPOLOGY
=========================================

Train RL for diamond:
TOPO_MODE=diamond python3 train_rl_agent.py
TOPO_MODE=diamond python3 plot_training_results.py

FIFO diamond:
Terminal 1:
cd ~/CompProject
sudo mn -c
source ~/ryu38/bin/activate
TOPO_MODE=diamond ryu-manager --verbose ryu_fifo_controller.py

Terminal 2:
cd ~/CompProject
TOPO_MODE=diamond EVAL_NUM_RUNS=5 EVAL_NUM_FLOWS=150 sudo -E python3 diamond_topology.py --policy FIFO

RL diamond:
Terminal 1:
cd ~/CompProject
sudo mn -c
source ~/ryu38/bin/activate
TOPO_MODE=diamond ryu-manager --verbose ryu_rl_controller.py

Terminal 2:
cd ~/CompProject
TOPO_MODE=diamond EVAL_NUM_RUNS=5 EVAL_NUM_FLOWS=150 sudo -E python3 diamond_topology.py --policy RL

Evaluate diamond:
TOPO_MODE=diamond python3 eval_fifo_vs_rl.py

=========================================
 THREE-PATH TOPOLOGY
=========================================

Train RL for three-path:
TOPO_MODE=three_path python3 train_rl_agent.py
TOPO_MODE=three_path python3 plot_training_results.py

FIFO three-path:
Terminal 1:
cd ~/CompProject
sudo mn -c
source ~/ryu38/bin/activate
TOPO_MODE=three_path ryu-manager --verbose ryu_fifo_controller.py

Terminal 2:
cd ~/CompProject
TOPO_MODE=three_path EVAL_NUM_RUNS=5 EVAL_NUM_FLOWS=150 sudo -E python3 three_path_topology.py --policy FIFO

RL three-path:
Terminal 1:
cd ~/CompProject
sudo mn -c
source ~/ryu38/bin/activate
TOPO_MODE=three_path ryu-manager --verbose ryu_rl_controller.py

Terminal 2:
cd ~/CompProject
TOPO_MODE=three_path EVAL_NUM_RUNS=5 EVAL_NUM_FLOWS=150 sudo -E python3 three_path_topology.py --policy RL

Evaluate three-path:
TOPO_MODE=three_path python3 eval_fifo_vs_rl.py

Important output files:
- data/q_table_diamond.json
- data/q_table_three_path.json
- data/training_state_visits_diamond.json
- data/training_state_visits_three_path.json
- data/diamond_fifo_vs_rl_summary.csv
- data/three_path_fifo_vs_rl_summary.csv
- data/diamond_fifo_vs_rl_summary_by_seed.csv
- data/three_path_fifo_vs_rl_summary_by_seed.csv
- data/plots/
EOF

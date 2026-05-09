#!/bin/bash

PROJECT_DIR="$HOME/CompProject"
RYU_ENV="$HOME/ryu38"

echo "========================================="
echo " Multipath SDN RL Project Instructions"
echo "========================================="
echo

echo "Professor workflow:"
echo "1. FIFO baseline with multipath topology"
echo "2. RL training only in Gym environment"
echo "3. Export Q-table / policy"
echo "4. Load policy into Ryu"
echo "5. Evaluate FIFO vs RL"
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
echo "- data/reward_curve_total.png"
echo "- data/reward_curve_average.png"
echo "- data/q_table_values.png"
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
echo "sudo python3 diamond_topology.py"
echo
echo "Inside Mininet:"
echo "pingall"
echo "h1 ping -c 20 h5"
echo "h1 ping -c 20 h6"
echo "h1 ping -c 50 h5 &"
echo "h2 ping -c 50 h6 &"
echo "h3 ping -c 50 h7 &"
echo "h4 ping -c 50 h8 &"
echo "sh ovs-ofctl -O OpenFlow13 dump-flows s1"
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
echo "sudo python3 diamond_topology.py"
echo
echo "Inside Mininet:"
echo "pingall"
echo "h1 ping -c 20 h5"
echo "h1 ping -c 20 h6"
echo "h1 ping -c 50 h5 &"
echo "h2 ping -c 50 h6 &"
echo "h3 ping -c 50 h7 &"
echo "h4 ping -c 50 h8 &"
echo "sh ovs-ofctl -O OpenFlow13 dump-flows s1"
echo
echo "RL decisions saved to data/rl_metrics.csv"
echo

echo "========================================="
echo " Step 5: Evaluate FIFO vs RL"
echo "========================================="
echo "cd $PROJECT_DIR"
echo "python3 eval_fifo_vs_rl.py"
echo

echo "========================================="
echo " Files to screenshot for report"
echo "========================================="
echo "- data/reward_curve_total.png"
echo "- data/reward_curve_average.png"
echo "- data/q_table_values.png"
echo "- pingall output"
echo "- dump-flows s1 output"
echo "- eval_fifo_vs_rl.py output"
echo

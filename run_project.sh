#!/bin/bash

PROJECT_DIR="$HOME/CompProject"
RYU_ENV="$HOME/ryu38"

echo "========================================="
echo " Multipath SDN RL Project Instructions"
echo "========================================="
echo

echo "Professor's step-by-step:"
echo "1. Use FIFO with multipath topology as baseline"
echo "2. Train RL only in Gym environment"
echo "3. Load trained agent/policy into Ryu"
echo "4. Evaluate FIFO vs RL"
echo

echo "========================================="
echo " Step 0: Optional cleanup"
echo "========================================="
echo "Run before starting a new demo:"
echo
echo "cd $PROJECT_DIR"
echo "sudo mn -c"
echo

echo "========================================="
echo " Step 1: Train RL agent offline"
echo "========================================="
echo "Run this first:"
echo
echo "cd $PROJECT_DIR"
echo "python3 train_rl_agent.py"
echo
echo "This creates:"
echo "- data/q_table.json"
echo "- data/training_rewards.csv"
echo

echo "========================================="
echo " Step 2: FIFO baseline demo"
echo "========================================="
echo
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
echo "h1 ping -c 10 h5"
echo "h1 ping -c 10 h6"
echo "h2 ping -c 10 h7"
echo "h3 ping -c 10 h8"
echo "sh ovs-ofctl -O OpenFlow13 dump-flows s1"
echo
echo "FIFO results saved to:"
echo "data/fifo_metrics.csv"
echo

echo "========================================="
echo " Step 3: RL controller demo"
echo "========================================="
echo
echo "Stop Mininet first:"
echo "exit"
echo
echo "Then stop Ryu with CTRL+C."
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
echo "h1 ping -c 10 h5"
echo "h1 ping -c 10 h6"
echo "h2 ping -c 10 h7"
echo "h3 ping -c 10 h8"
echo "sh ovs-ofctl -O OpenFlow13 dump-flows s1"
echo
echo "RL results saved to:"
echo "data/rl_metrics.csv"
echo

echo "========================================="
echo " Step 4: Evaluate"
echo "========================================="
echo
echo "After running both FIFO and RL:"
echo
echo "cd $PROJECT_DIR"
echo "python3 eval_fifo_vs_rl.py"
echo

echo "========================================="
echo " What to say in the demo"
echo "========================================="
echo
echo "The topology is a fixed diamond multipath topology."
echo "FIFO is our baseline controller."
echo "The RL agent is trained offline in a Gym-style environment."
echo "Ryu does not train the model."
echo "Ryu only loads the learned policy and installs OpenFlow rules."
echo "We evaluate FIFO vs RL using path usage, connectivity, and flow logs."
echo

#!/bin/bash

PROJECT_DIR="$HOME/CompProject"
RYU_ENV="$HOME/ryu38"

echo "========================================="
echo " Multipath SDN Project Instructions"
echo "========================================="
echo

echo "[1] First terminal (Ryu Controller)"
echo "-----------------------------------"
echo "cd $PROJECT_DIR"
echo "sudo mn -c"
echo "source $RYU_ENV/bin/activate"
echo "ryu-manager --verbose ryu_multipath_controller.py"
echo

echo "[2] Second terminal (Mininet Topology)"
echo "--------------------------------------"
echo "cd $PROJECT_DIR"
echo "sudo python3 diamond_topology.py"
echo

echo "[3] Inside Mininet"
echo "------------------"
echo "pingall"
echo "h1 ping -c 3 h5"
echo "h1 ping -c 3 h6"
echo "sh ovs-ofctl -O OpenFlow13 dump-flows s1"
echo
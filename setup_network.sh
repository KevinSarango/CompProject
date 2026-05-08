#!/bin/bash
# setup_network.sh
# Cleans up leftover Mininet/OVS processes and prepares the environment

echo Terminal 1 — full cleanup
sudo ovs-vsctl list-br | xargs -r -I{} sudo ovs-vsctl del-br {}
sudo ip netns list | awk '{print $1}' | xargs -r -I{} sudo ip netns delete {}
sudo pkill -f mininet 2>/dev/null
sudo pkill -f ryu 2>/dev/null
sudo fuser -k 6653/tcp 2>/dev/null
sudo systemctl restart openvswitch-switch
sleep 2

# Confirm clean
sudo ovs-vsctl show

echo "[INFO] Stopping any running Mininet and OVS processes..."
sudo pkill -f mininet
sudo pkill -f ovs

echo "[INFO] Deleting leftover network namespaces..."
sudo ip netns | xargs -r -n1 sudo ip netns delete

echo "[INFO] Deleting leftover OVS bridges..."
sudo ovs-vsctl list-br | xargs -r sudo ovs-vsctl del-br

echo "[INFO] Deleting leftover Mininet virtual interfaces..."
ip link show | grep -oP '(?<=\d: )[sd]\d+[-\w]*(?=@|:)' | while read intf; do
    echo "  Deleting $intf"
    sudo ip link delete "$intf" 2>/dev/null
done

echo "[INFO] Restarting Open vSwitch..."
sudo systemctl restart openvswitch-switch

# Optional: show status
echo "[INFO] Current network namespaces:"
ip netns
echo "[INFO] Current OVS bridges:"
sudo ovs-vsctl list-br
echo "[INFO] Current virtual interfaces (veths):"
ip link | grep -E 's[0-9]+|d[0-9]+'

echo "[INFO] Cleanup complete!"
echo "Next steps:"
echo "1) Start your Mininet topology in a new terminal:"
echo "   cd /home/vboxuser/CompProject"
echo "   source /home/vboxuser/comp/bin/activate"
echo "   export PYTHONPATH=\$PYTHONPATH:/home/vboxuser/CompProject"
echo "   sudo /home/vboxuser/comp/bin/python3 my_topology.py"
echo "2) Start your Ryu controller in a new terminal:"
echo "   cd /home/vboxuser/CompProject"
echo "   source /home/vboxuser/comp/bin/activate"
echo "   export PYTHONPATH=\$PYTHONPATH:/home/vboxuser/CompProject"
echo "   ryu-manager --ofp-tcp-listen-port 6653 ryu_controller.py"
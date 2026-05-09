#!/bin/bash

echo "Run these commands INSIDE the Mininet CLI."
echo
echo "Basic connectivity:"
echo "pingall"
echo
echo "Sequential traffic:"
echo "h1 ping -c 20 h5"
echo "h1 ping -c 20 h6"
echo "h2 ping -c 20 h7"
echo "h3 ping -c 20 h8"
echo
echo "Parallel traffic burst:"
echo "h1 ping -c 50 h5 &"
echo "h2 ping -c 50 h6 &"
echo "h3 ping -c 50 h7 &"
echo "h4 ping -c 50 h8 &"
echo
echo "Cross traffic burst:"
echo "h1 ping -c 50 h8 &"
echo "h2 ping -c 50 h7 &"
echo "h3 ping -c 50 h6 &"
echo "h4 ping -c 50 h5 &"
echo
echo "Optional iperf test:"
echo "iperf h1 h5"
echo "iperf h2 h6"
echo "iperf h3 h7"
echo "iperf h4 h8"
echo
echo "Inspect paths:"
echo "sh ovs-ofctl -O OpenFlow13 dump-flows s1"
echo "sh ovs-ofctl -O OpenFlow13 dump-flows s2"
echo "sh ovs-ofctl -O OpenFlow13 dump-flows s3"
echo "sh ovs-ofctl -O OpenFlow13 dump-flows s4"

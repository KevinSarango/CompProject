#!/bin/bash

echo "Run these inside the Mininet CLI:"
echo
echo "pingall"
echo "h1 ping -c 10 h5"
echo "h1 ping -c 10 h6"
echo "h2 ping -c 10 h7"
echo "h3 ping -c 10 h8"
echo
echo "For parallel traffic:"
echo "h1 ping -c 20 h5 &"
echo "h2 ping -c 20 h6 &"
echo "h3 ping -c 20 h7 &"
echo "h4 ping -c 20 h8 &"
echo
echo "To inspect paths:"
echo "sh ovs-ofctl -O OpenFlow13 dump-flows s1"
echo "sh ovs-ofctl -O OpenFlow13 dump-flows s2"
echo "sh ovs-ofctl -O OpenFlow13 dump-flows s3"
echo "sh ovs-ofctl -O OpenFlow13 dump-flows s4"

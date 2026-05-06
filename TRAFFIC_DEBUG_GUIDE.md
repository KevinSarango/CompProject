# Traffic Flow Debugging Guide

## Changes Made

### 1. **ryu_controller.py** — Force all traffic to controller

**Problem**: The original code only sent unmatched traffic to the controller. Once a flow was installed, it never went to the controller again.

**Solution**: 
- Changed switch_features_handler to install a **high-priority catch-all rule (priority 65535)** that sends ALL traffic to the controller
- This ensures every packet is visible to the controller for monitoring
- Added verbose logging at every stage: PACKET_IN, FLOW_STATS, SWITCH_CONNECT

**Result**: Now you'll see:
```
[SWITCH] Connected: dpid=3
[SWITCH] Installed catch-all rule on dpid=3 to send all traffic to controller
[PACKET_IN] dpid=3 in_port=1 buffer_id=256
[PACKET_IN] src=00:00:00:00:00:01 dst=00:00:00:00:00:02
[PACKET_IN] Installed flow for 00:00:00:00:00:02 out on port 2
[MONITOR] switches=2 | flows=8 | stats_update=1.2ms
```

### 2. **my_topology.py** — Better logging and traffic startup

**Problems**: 
- Couldn't see if traffic was actually being generated
- Hosts might not be found, traffic might fail silently
- No indication of progress

**Solutions**:
- Added detailed logging in `gen_iperf_flow()` showing what's being started
- Added host existence verification in `launch_bottleneck_traffic()`
- Increased wait times between iperf server startup and client connection
- Added network topology printout showing controller address
- Better traffic scenario logging

**Result**: Now you'll see:
```
[TRAFFIC] Launching 4 competing flows on bottleneck...
[TRAFFIC] Available hosts: ['src1', 'dst1', 'src2', 'dst2', ...]
[IPERF] Starting bulk flow: src1 -> dst1:5200
[IPERF] dst1: iperf3 -s -p 5200 -D ...
[IPERF] src1: iperf3 -c 10.0.0.11 -p 5200 -t 120 -b 50M ... &
```

---

## Verification Steps

### Step 1: Clean Network (run in Terminal 1)
```bash
cd /home/vboxuser/CompProject
source /home/vboxuser/comp/bin/activate
export PYTHONPATH=$PYTHONPATH:/home/vboxuser/CompProject
sudo bash setup_network.sh
```

### Step 2: Start Ryu Controller (Terminal 2)
```bash
cd /home/vboxuser/CompProject
source /home/vboxuser/comp/bin/activate
export PYTHONPATH=$PYTHONPATH:/home/vboxuser/CompProject
ryu-manager --ofp-tcp-listen-port 6653 ryu_controller.py
```

**Expected output**:
```
loaded app ryu.controller.ofp_handler
loaded app ryu_controller.ClosedLoopController
[MONITOR] Waiting for switches to connect...
[RL] Waiting for switches to connect...
```

### Step 3: Start Mininet Topology (Terminal 3)
```bash
cd /home/vboxuser/CompProject
source /home/vboxuser/comp/bin/activate
export PYTHONPATH=$PYTHONPATH:/home/vboxuser/CompProject
sudo /home/vboxuser/comp/bin/python3 my_topology.py 4
```

**Expected output in Mininet terminal**:
```
============================================================
*** Bottleneck Network Started ***
*** Switches : 2 core switches with 10 Mbps bottleneck link
*** Hosts    : 8 (4 sources + 4 destinations)
*** Flows    : 4 competing flows
*** Controller: 127.0.0.1:6653
============================================================

[SETUP] Waiting for all switches to connect to controller...
[SETUP] Testing connectivity with ping...
*** Ping: testing ping reachability
*** Results: 0% packet loss (8 received)
[SETUP] Connectivity verified!
[SETUP] Starting traffic generation in background thread...

[TRAFFIC] Launching 4 competing flows on bottleneck...
[TRAFFIC] Available hosts: ['src1', 'dst1', 'src2', 'dst2', ...]
[IPERF] Starting bulk flow: src1 -> dst1:5200
```

**Expected output in Ryu terminal** (once traffic starts):
```
[SWITCH] Connected: dpid=3
[SWITCH] Installed catch-all rule on dpid=3 to send all traffic to controller
[SWITCH] Connected: dpid=4
[SWITCH] Installed catch-all rule on dpid=4 to send all traffic to controller
[PACKET_IN] dpid=3 in_port=1 buffer_id=256
[PACKET_IN] src=00:00:00:00:00:01 dst=00:00:00:00:00:02
[PACKET_IN] Installed flow for 00:00:00:00:00:02 out on port 2
[PACKET_IN] Sent PacketOut to port 2
[MONITOR] switches=2 | flows=4 | stats_update=1.5ms
[MONITOR] switches=2 | flows=8 | stats_update=2.1ms
```

---

## Troubleshooting

### Issue: "No switches connected" in Ryu
**Causes**:
1. Mininet topology not started
2. Controller port 6653 in use
3. RemoteController address wrong

**Fix**:
```bash
# Check if port 6653 is in use
sudo lsof -i :6653

# Check if Mininet can reach controller
sudo python3 -c "import socket; s=socket.socket(); s.connect(('127.0.0.1', 6653)); print('OK')"
```

### Issue: No PACKET_IN events in Ryu
**Causes**:
1. Catch-all rule not installed (check logs for "[SWITCH] Installed catch-all rule")
2. Traffic not being generated
3. iperf3 not installed on hosts

**Fix**:
```bash
# In Mininet CLI
src1 iperf3 --version
dst1 iperf3 --version

# Or check if traffic processes exist
src1 ps aux | grep iperf
```

### Issue: Flows show 0 packets in MONITOR
**Causes**:
1. Traffic iperf connections failing
2. Flows matching but no actual packets

**Fix**:
```bash
# In Mininet CLI, test direct connectivity
src1 ping -c 3 dst1

# Check if iperf is listening on dst
dst1 ss -tlnp | grep 5200

# Check iperf logs
src1 cat /tmp/bulk_src1.log
```

### Issue: Traffic generation hangs
**Causes**:
1. iperf3 server not starting
2. Firewall blocking connections

**Fix**:
```bash
# Manually start and test
dst1 iperf3 -s -p 5200
# In another host
src1 timeout 5 iperf3 -c 10.0.0.11 -p 5200 -t 10
```

---

## What's Happening

### Control Plane (Ryu → Mininet)
1. **Switch Connect**: Mininet switches (s3, s4) connect to Ryu controller on 127.0.0.1:6653
2. **Catch-All Rule**: Ryu installs rule on each switch to send ALL traffic to controller
3. **Packet-In**: Every packet triggers a packet-in event (except after flow install)
4. **Flow Install**: For known destination MACs, Ryu installs hardware flows to forward directly (lower priority than catch-all)

### Data Plane (Mininet)
1. **Traffic Generation**: iperf3 processes send UDP/TCP traffic
2. **Switch Processing**: Traffic hits s3, matches catch-all rule, goes to controller AND out ports
3. **Bottleneck Link**: s3-s4 link at 10 Mbps causes congestion
4. **Statistics**: Controller polls for flow stats every 1 second

---

## Monitoring in Real-Time

### Terminal 4: Watch Ryu logs
```bash
tail -f /tmp/ryu_controller.log 2>/dev/null || journalctl -u ryu -f
```

### Terminal 5: Watch metrics
```bash
watch -n 0.5 'tail -5 /home/vboxuser/CompProject/data/rl_metrics_log.csv'
```

### In Mininet CLI: Check active flows
```bash
mininet> s3 ovs-ofctl dump-flows s3
mininet> s4 ovs-ofctl dump-flows s4
```

### Check packet counts in real-time
```bash
mininet> s3 ovs-ofctl dump-table-stats s3
```

---

## Expected Performance

With the catch-all rule and traffic:
- **Packet-in rate**: ~100-1000 per second (first packet of each microflow)
- **Flow count**: 4-8 active flows
- **Stats collection time**: 1-3ms per poll
- **Bottleneck saturation**: 10 Mbps across ~4 competing flows

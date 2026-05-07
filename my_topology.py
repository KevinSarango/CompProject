"""
bottleneck_topology.py

Simple bottleneck topology for RL QoS training.
Multiple traffic types compete at a congested link.
Much faster training than enterprise campus topology.
"""

from mininet.net import Mininet
from mininet.node import RemoteController
from mininet.cli import CLI
from mininet.link import TCLink
import time
import threading

def gen_iperf_flow(src, dst, port, flow_type, duration=120):
    """Generate iperf flow with traffic-specific parameters."""
    # print(f"\n[IPERF] Starting {flow_type} flow: {src.name} -> {dst.name}:{port}")
    
    # Start server
    srv_cmd = f'iperf3 -s -p {port} -D --logfile /tmp/iperf_{dst.name}_{port}.log'
    # print(f"[IPERF] {dst.name}: {srv_cmd}")
    dst.cmd(srv_cmd)
    time.sleep(0.5)
    
    # Start client with traffic-specific settings
    if flow_type == 'bulk':
        cmd = (f'iperf3 -c {dst.IP()} -p {port} -t {duration} -b 50M '
               f'--logfile /tmp/bulk_{src.name}.log')
    elif flow_type == 'video':
        cmd = (f'iperf3 -c {dst.IP()} -p {port} -u -b 5M -l 1000 -t {duration} '
               f'--logfile /tmp/video_{src.name}.log')
    elif flow_type == 'voip':
        cmd = (f'iperf3 -c {dst.IP()} -p {port} -u -b 64k -l 200 -t {duration} '
               f'--logfile /tmp/voip_{src.name}.log')
    elif flow_type == 'interactive':
        cmd = (f'iperf3 -c {dst.IP()} -p {port} -b 1M -t {duration} '
               f'--logfile /tmp/interactive_{src.name}.log')
    
    # print(f"[IPERF] {src.name}: {cmd} &")
    src.cmd(f'{cmd} &')
    # print(f"[IPERF] {flow_type} flow started: {src.name} -> {dst.name}:{port}")

def launch_bottleneck_traffic(net, num_flows=4, duration=120):
    """
    Launch competing flows across the bottleneck.
    num_flows: how many concurrent flows (default 4)
    Logs traffic scenario to traffic_scenario.log for RL correlation analysis.
    """
    hosts = {h.name: h for h in net.hosts}
    
    # print(f"\n{'='*60}")
    # print(f"[TRAFFIC] Launching {num_flows} competing flows on bottleneck...")
    # print(f"[TRAFFIC] Duration: {duration}s")
    # print(f"[TRAFFIC] Available hosts: {list(hosts.keys())}")
    # print(f"{'='*60}\n")
    
    flow_types = ['bulk', 'video', 'voip', 'interactive']
    port = 5200
    
    # Log traffic scenario for correlation analysis
    with open('traffic_scenario.log', 'w') as f:
        f.write('timestamp,src,dst,flow_type,bitrate_mbps,protocol,packet_size\n')
        
        for i in range(num_flows):
            src_name = f'src{i+1}'
            dst_name = f'dst{i+1}'
            flow_type = flow_types[i % len(flow_types)]
            
            src = hosts.get(src_name)
            dst = hosts.get(dst_name)
            
            if src and dst:
                # Define traffic characteristics
                if flow_type == 'bulk':
                    bitrate, protocol, pkt_size = 50, 'TCP', 'MTU'
                elif flow_type == 'video':
                    bitrate, protocol, pkt_size = 5, 'UDP', '1000B'
                elif flow_type == 'voip':
                    bitrate, protocol, pkt_size = 0.064, 'UDP', '200B'
                elif flow_type == 'interactive':
                    bitrate, protocol, pkt_size = 1, 'TCP', 'mixed'
                
                # Log this traffic flow
                f.write(f'{time.time()},{src_name},{dst_name},{flow_type},'
                        f'{bitrate},{protocol},{pkt_size}\n')
                
                print(f"  [{flow_type.upper():12s}] {src_name} -> {dst_name}")
                gen_iperf_flow(src, dst, port, flow_type, duration)
                port += 1
                time.sleep(0.5)   # increased stagger for better startup
            else:
                print(f"[TRAFFIC] WARNING: Could not find hosts {src_name} and/or {dst_name}")
    
    print(f"\n[TRAFFIC] {num_flows} flows started. Competing at bottleneck link.")
    print(f"[TRAFFIC] Scenario logged to traffic_scenario.log\n")
    time.sleep(2)
    
    # Verify traffic is running
    print("[TRAFFIC] Verifying traffic is running...")
    time.sleep(5)
    print("[TRAFFIC] Traffic generation complete.")

# ------------------ Bottleneck Topology ------------------

from mininet.topo import Topo

class BottleneckTopo(Topo):
    """
    Simple bottleneck topology:
    - N source hosts
    - N destination hosts
    - All traffic forced through core switch (s3) → core switch (s4)
    - Congested link: s3-s4 bottleneck
    """
    def __init__(self, num_flows=4, **opts):
        super().__init__(**opts)

        # Store number of flows
        self.num_flows = num_flows

        # Core switches (bottleneck)
        s3 = self.addSwitch('s3', dpid='0000000000000003')
        s4 = self.addSwitch('s4', dpid='0000000000000004')

        # Bottleneck link
        self.addLink(s3, s4, bw=10)  # 10 Mbps bottleneck

        # Source hosts connect to s3
        for i in range(1, num_flows + 1):
            src = self.addHost(f'src{i}', ip=f'10.0.0.{i}/24')
            self.addLink(src, s3, bw=100)

        # Destination hosts connect to s4
        for i in range(1, num_flows + 1):
            dst = self.addHost(f'dst{i}', ip=f'10.0.0.{i+10}/24')
            self.addLink(dst, s4, bw=100)

# ------------------ Run Network ------------------

def run(num_flows=4):

    topo = BottleneckTopo(num_flows=num_flows)
    net = Mininet(
        topo=topo,
        controller=lambda name: RemoteController(name, ip='127.0.0.1', port=6653),
        link=TCLink
    )

    try:
        net.start()
        print("\n" + "="*60)
        print("*** Bottleneck Network Started ***")
        print(f"*** Switches : 2 core switches with 10 Mbps bottleneck link")
        print(f"*** Hosts    : {num_flows*2} ({num_flows} sources + {num_flows} destinations)")
        print(f"*** Flows    : {num_flows} competing flows")
        print(f"*** Controller: 127.0.0.1:6653")
        print("="*60 + "\n")
        
        print("[SETUP] Waiting for all switches to connect to controller...")
        time.sleep(3)

        print("[SETUP] Testing connectivity with ping...")
        net.pingAll()
        time.sleep(2)
        
        print("[SETUP] Second ping test...")
        net.pingAll()
        
        print("[SETUP] Connectivity verified!")
        print("[SETUP] Starting traffic generation in background thread...\n")
        
        # Launch competing traffic in a thread
        t = threading.Thread(
            target=launch_bottleneck_traffic,
            args=(net,),
            kwargs={'num_flows': num_flows, 'duration': 180},
            daemon=True
        )
        t.start()

        print("[SETUP] Entering CLI - type 'exit' to stop\n")
        CLI(net)
    finally:
        net.stop()
        print("\n*** Network stopped and cleaned up ***")

# ------------------ Main ------------------

if __name__ == '__main__':
    import sys
    num_flows = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    run(num_flows)
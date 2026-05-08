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
from mininet.topo import Topo
import time
import threading


def gen_iperf_flow(src, dst, port, flow_type, duration=120):
    """Generate iperf flow with traffic-specific parameters."""
    # Start server
    srv_cmd = f'iperf3 -s -p {port} -D --logfile /tmp/iperf_{dst.name}_{port}.log'
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

    src.cmd(f'{cmd} &')


def configure_queues(net):
    """Configure QoS queues on all core and bottleneck links."""
    print("[SETUP] Configuring QoS queues on all switches...")

    switches = ['s1', 's2', 's3', 's4', 's5', 's6']
    
    for sw_name in switches:
        sw = net.get(sw_name)
        if not sw:
            continue
        
        # Get all outgoing ports (skip local port 65534)
        ports = [str(i) for i in range(1, 10)]  # Assume max 9 ports per switch
        
        for port_num in ports:
            port_name = f'{sw_name}-eth{port_num}'
            
            try:
                # Configure queues on this port
                sw.cmd(f'ovs-vsctl -- set Port {sw_name}-eth{port_num} qos=@newqos -- '
                       '--id=@newqos create QoS type=linux-htb other-config:max-rate=10000000 '
                       'queues=0=@q0,1=@q1,2=@q2,3=@q3,4=@q4 -- '
                       '--id=@q0 create Queue other-config:min-rate=10000000 other-config:max-rate=10000000 -- '
                       '--id=@q1 create Queue other-config:min-rate=7500000 other-config:max-rate=7500000 -- '
                       '--id=@q2 create Queue other-config:min-rate=5000000 other-config:max-rate=5000000 -- '
                       '--id=@q3 create Queue other-config:min-rate=2500000 other-config:max-rate=2500000 -- '
                       '--id=@q4 create Queue other-config:min-rate=1000000 other-config:max-rate=1000000')
            except Exception as e:
                pass  # Port may not exist, skip silently
    
    print("[SETUP] QoS queues configured on all switches: "
          "queue 0=10Mbps, 1=7.5Mbps, 2=5Mbps, 3=2.5Mbps, 4=1Mbps")

def launch_bottleneck_traffic(net, num_flows=4, duration=3600):
    """
    Launch competing flows across the bottleneck.
    Uses the 4 hosts in the topology.
    num_flows: how many flows to generate between hosts (default 4)
    """
    hosts = {h.name: h for h in net.hosts}

    print(f"\n[TRAFFIC] Launching {num_flows} competing flows on bottleneck...")
    print(f"[TRAFFIC] Duration: {duration}s (covers full training)\n")

    flow_types = ['bulk', 'video', 'voip', 'interactive']
    port = 5200
    
    # Generate flows between all pairs of hosts
    host_list = ['h1', 'h2', 'h3', 'h4']
    
    flow_count = 0
    for i, src_name in enumerate(host_list):
        for dst_name in host_list[i+1:]:
            if flow_count >= num_flows:
                break
            
            flow_type = flow_types[flow_count % len(flow_types)]
            src = hosts.get(src_name)
            dst = hosts.get(dst_name)

            if src and dst:
                print(f"  [{flow_type.upper():12s}] {src_name} -> {dst_name}")
                gen_iperf_flow(src, dst, port, flow_type, duration)
                port += 1
                flow_count += 1
                time.sleep(0.5)
        
        if flow_count >= num_flows:
            break

    print(f"\n[TRAFFIC] {flow_count} flows started. Competing on multiple bottleneck links.")


# ------------------ Multi-Path Bottleneck Topology ------------------

class BottleneckTopo(Topo):
    """Multi-path bottleneck topology with 4 switches forming a chain + 2 core switches."""
    def __init__(self, num_flows=4, **opts):
        super().__init__(**opts)

        # Create 4 edge switches connected in a chain
        s1 = self.addSwitch('s1', dpid='0000000000000001')
        s2 = self.addSwitch('s2', dpid='0000000000000002')
        s3 = self.addSwitch('s3', dpid='0000000000000003')
        s4 = self.addSwitch('s4', dpid='0000000000000004')

        # Create 2 core switches
        s5 = self.addSwitch('s5', dpid='0000000000000005')
        s6 = self.addSwitch('s6', dpid='0000000000000006')

        # Add 4 hosts
        h1 = self.addHost('h1', ip='10.0.0.1/24')
        h2 = self.addHost('h2', ip='10.0.0.2/24')
        h3 = self.addHost('h3', ip='10.0.0.3/24')
        h4 = self.addHost('h4', ip='10.0.0.4/24')

        # Connect hosts to their corresponding switches (100 Mbps access links)
        self.addLink(h1, s1, bw=100)
        self.addLink(h2, s2, bw=100)
        self.addLink(h3, s3, bw=100)
        self.addLink(h4, s4, bw=100)

        # Connect switches as specified:
        # s1 connects to s2 and s5
        self.addLink(s1, s2, bw=10)
        self.addLink(s1, s5, bw=10)

        # s2 connects to s1 and s3 (s1 already connected above)
        self.addLink(s2, s3, bw=10)

        # s3 connects to s2 and s4 (s2 already connected above)
        self.addLink(s3, s4, bw=10)

        # s4 connects to s3 and s6 (s3 already connected above)
        self.addLink(s4, s6, bw=10)

        # s5 connects to s1 and s6 (s1 already connected above)
        self.addLink(s5, s6, bw=10)

        # s6 connects to s5 and s4 (both already connected above)


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
        print("*** Multi-Path Bottleneck Network Started ***")
        print("*** Switches  : 6 (s1-s4 chain + s5-s6 core)")
        print("*** Hosts     : 4 (h1→s1, h2→s5, h3→s6, h4→s4)")
        print("*** Bottlenecks: 5 × 10 Mbps links")
        print("***   Path 1: s1↔s2↔s3↔s4 (primary chain)")
        print("***   Path 2: s1↔s5↔s6↔s4 (alternate core path)")
        print("*** Flows     : Multi-path routing possible")
        print("*** Controller: 127.0.0.1:6653")
        print("="*60 + "\n")

        print("[SETUP] Waiting for switches to connect to controller...")
        time.sleep(3)

        print("[SETUP] Testing connectivity with pingall...")
        net.pingAll()
        time.sleep(2)

        print("[SETUP] Second ping to fully populate MAC tables...")
        net.pingAll()

        print("[SETUP] Connectivity verified!")
        configure_queues(net)

        print("[SETUP] Starting traffic generation in background thread...\n")
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
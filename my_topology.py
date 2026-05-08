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
import os


def gen_iperf_flow(src, dst, port, flow_type, duration=120):
    """Generate iperf flow with traffic-specific parameters."""
    # Ensure no stale iperf3 server is running on this port, then start server
    dst.cmd(f'pkill -f "iperf3 -s -p {port}" 2>/dev/null')
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
    else:
        # Unknown flow type — skip
        return

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
    def __init__(self, **opts):
        super().__init__(**opts)

        s1 = self.addSwitch('s1', dpid='0000000000000001')
        s2 = self.addSwitch('s2', dpid='0000000000000002')
        s3 = self.addSwitch('s3', dpid='0000000000000003')
        s4 = self.addSwitch('s4', dpid='0000000000000004')

        s5 = self.addSwitch('s5', dpid='0000000000000005')
        s6 = self.addSwitch('s6', dpid='0000000000000006')

        h1 = self.addHost('h1', ip='10.0.0.1/24')
        h2 = self.addHost('h2', ip='10.0.0.2/24')
        h3 = self.addHost('h3', ip='10.0.0.3/24')
        h4 = self.addHost('h4', ip='10.0.0.4/24')

        self.addLink(h1, s1, bw=100)
        self.addLink(h2, s2, bw=100)
        self.addLink(h3, s3, bw=100)
        self.addLink(h4, s4, bw=100)

        self.addLink(s1, s2, bw=10)
        self.addLink(s1, s5, bw=10)
        self.addLink(s2, s3, bw=10)
        self.addLink(s3, s4, bw=10)
        self.addLink(s4, s6, bw=10)
        self.addLink(s5, s6, bw=10)


# ------------------ Run Network ------------------

def run():
    topo = BottleneckTopo()
    net = Mininet(
        topo=topo,
        controller=lambda name: RemoteController(name, ip='127.0.0.1', port=6653),
        link=TCLink
    )

    try:
        net.start()

        print("\n" + "="*70)
        print("[SETUP] Network started. Waiting for switches to connect...")
        print("="*70)
        time.sleep(5)

        print("\n[SETUP] Testing connectivity with pingall...")
        # Use a small timeout to keep connectivity checks fast
        result1 = net.pingAll(timeout=2)
        time.sleep(2)

        print("\n[SETUP] Second ping to fully populate MAC tables...")
        result2 = net.pingAll(timeout=2)
        
        print("\n[SETUP] Checking ping results...")
        time.sleep(2)

        # Verify all pings succeeded
        if result1 != 0.0 or result2 != 0.0:
            print("\n[WARNING] Not all pings succeeded!")
            print(f"  First pingAll packet loss: {result1*100:.1f}%")
            print(f"  Second pingAll packet loss: {result2*100:.1f}%")
            print("[WARNING] Retrying connectivity check...")
            
            # Retry up to 3 times
            for attempt in range(3):
                print(f"\n[SETUP] Retry {attempt + 1}/3...")
                time.sleep(2)
                result = net.pingAll()
                print(f"  Packet loss: {result*100:.1f}%")
                if result == 0.0:
                    print("[SUCCESS] All hosts can reach each other!")
                    break
            else:
                print("\n[ERROR] Connectivity check failed after 3 retries!")
                print("[ERROR] Some hosts cannot reach each other. Check your topology.")
                net.stop()
                return

        print("\n[SETUP] ✓ Connectivity check complete - all hosts can reach each other!")

        # Create a signal file so external processes (e.g. the Ryu controller)
        # can detect that connectivity has been verified and proceed.
        try:
            signal_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'connectivity_verified.txt')
            with open(signal_path, 'w') as sf:
                sf.write('OK')
            print(f"[SETUP] Connectivity signal written: {signal_path}")
        except Exception as e:
            print(f"[WARN] Failed to write connectivity signal: {e}")
        configure_queues(net)

        print("\n" + "="*70)
        print("[SETUP] *** ENTERING MININET CLI ***")
        print("[SETUP] The Ryu controller is now in MAC LEARNING PHASE (30 seconds)")
        print("[SETUP] During this phase, run 'pingall' a few times to ensure")
        print("[SETUP] all hosts can reach each other:")
        print("[SETUP]   mininet> pingall")
        print("[SETUP]   mininet> pingall")
        print("[SETUP]")
        print("[SETUP] After 30 seconds, the RL agent will enable routing rules.")
        print("[SETUP] Type 'exit' to stop the network.")
        print("="*70 + "\n")
        
        CLI(net)

    finally:
        net.stop()
        print("\n*** Network stopped and cleaned up ***")


# ------------------ Main ------------------

if __name__ == '__main__':
    run()
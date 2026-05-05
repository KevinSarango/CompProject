"""
campus_topology.py

Enterprise Campus Topology with role-based hosts and realistic
application traffic simulation. Optimized to maintain campus
realism while keeping active flows under 30 for RL training
efficiency.

Topology changes from full version:
  - 2 hosts per access switch instead of 8 (20 total vs 40)
  - 1 traffic scenario per zone instead of 2-3
  - Core services still present and active
  - All 5 campus zones still represented

Traffic types per zone (one representative flow each):
  dist1 Academic  -> video      (lecture capture)
  dist2 Dormitory -> bulk       (file download)
  dist3 Admin     -> voip       (IP phone call)
  dist4 DC access -> bulk       (server backup)
  dist5 Library   -> interactive(web browsing)
  core services   -> background (health checks)
"""

from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import RemoteController
from mininet.cli import CLI
from mininet.link import TCLink
import time
import threading


# -------------------------------------------------------------------
# Traffic generation functions
# -------------------------------------------------------------------

def start_iperf_server(host, port):
    host.cmd(f'iperf3 -s -p {port} -D --logfile /tmp/iperf_{host.name}_{port}.log')
    time.sleep(0.3)


def gen_bulk_transfer(src, dst, port=5101, duration=60):
    """File server backup / NFS — high throughput TCP near MTU."""
    start_iperf_server(dst, port)
    src.cmd(f'iperf3 -c {dst.IP()} -p {port} -t {duration} -b 100M '
            f'--logfile /tmp/bulk_{src.name}.log &')


def gen_video_stream(src, dst, port=5102, duration=60):
    """Lecture capture / IP camera — steady UDP ~5Mbps."""
    start_iperf_server(dst, port)
    src.cmd(f'iperf3 -c {dst.IP()} -p {port} -u -b 5M -l 1000 -t {duration} '
            f'--logfile /tmp/video_{src.name}.log &')


def gen_voip(src, dst, port=5103, duration=60):
    """IP phone call G.711 — tiny UDP ~64kbps ~50pps."""
    start_iperf_server(dst, port)
    src.cmd(f'iperf3 -c {dst.IP()} -p {port} -u -b 64k -l 200 -t {duration} '
            f'--logfile /tmp/voip_{src.name}.log &')


def gen_interactive(src, dst, port=5104, duration=60):
    """HTTP/SSH/database — bursty low-bandwidth TCP."""
    start_iperf_server(dst, port)
    src.cmd(f'iperf3 -c {dst.IP()} -p {port} -b 1M -t {duration} '
            f'--logfile /tmp/interactive_{src.name}.log &')


def gen_background(src, dst, count=30, interval=0.5):
    """DNS/ARP/NTP/SNMP — low-rate ICMP ping."""
    src.cmd(f'ping -c {count} -i {interval} {dst.IP()} '
            f'> /tmp/bg_{src.name}.log 2>&1 &')


# -------------------------------------------------------------------
# Reduced traffic scenario — one flow per zone, all 5 types covered
# -------------------------------------------------------------------

def launch_campus_traffic(net, duration=120):
    """
    Launches one representative flow per campus zone so every
    traffic class (video, bulk, voip, interactive, background)
    is present without flooding the controller with flows.

    Zone → Traffic type mapping:
      Academic  (dist1) → VIDEO        lecture stream
      Dormitory (dist2) → BULK         large file download
      Admin     (dist3) → VOIP         IP phone call
      DC access (dist4) → BULK         server backup
      Library   (dist5) → INTERACTIVE  web browsing
      Core svcs         → BACKGROUND   health checks
    """

    hosts = {h.name: h for h in net.hosts}

    def get(name):
        return hosts.get(name)

    print("\n[TRAFFIC] Launching campus traffic (reduced flow mode)...")
    print(f"[TRAFFIC] Duration: {duration}s | Target: <30 active flows\n")

    scenarios = [
        # (src,    dst,         type,          description)

        # One flow per zone — covers all 5 traffic classes
        ('h11', 'h12',      'video',       'Academic:  lecture capture'),
        ('h21', 'h22',      'bulk',        'Dormitory: file download'),
        ('h31', 'h32',      'voip',        'Admin:     IP phone call'),
        ('h41', 'h42',      'bulk',        'DC:        server backup'),
        ('h51', 'h52',      'interactive', 'Library:   web browsing'),

        # Core service health checks — background class
        ('h11', 'cache',    'background',  'Core: cache health check'),
        ('h41', 'ipam',     'background',  'Core: IPAM poll'),
        ('h31', 'wifi_ctrl','background',  'Core: WiFi controller poll'),
        ('h21', 'nms',      'background',  'Core: NMS SNMP poll'),
    ]

    port = 5200
    for src_name, dst_name, ttype, label in scenarios:
        src = get(src_name)
        dst = get(dst_name)
        if src is None or dst is None:
            print(f"[TRAFFIC] Skipping '{label}' — host not found")
            continue

        print(f"  [{ttype.upper():12s}] {src_name:8s} -> {dst_name:10s}  ({label})")

        if ttype == 'bulk':
            gen_bulk_transfer(src, dst, port=port, duration=duration)
        elif ttype == 'video':
            gen_video_stream(src, dst, port=port, duration=duration)
        elif ttype == 'voip':
            gen_voip(src, dst, port=port, duration=duration)
        elif ttype == 'interactive':
            gen_interactive(src, dst, port=port, duration=duration)
        elif ttype == 'background':
            gen_background(src, dst)

        port += 1
        time.sleep(0.2)

    print(f"\n[TRAFFIC] {len(scenarios)} flows started across 5 campus zones.")
    print("[TRAFFIC] All 5 traffic classes represented.")
    print("[TRAFFIC] Ryu RL controller classifying and applying QoS.\n")


# -------------------------------------------------------------------
# Topology — reduced hosts (2 per access switch instead of 8)
# All 5 zones + core services still present
# -------------------------------------------------------------------

class EnterpriseCampusTopo(Topo):
    def build(self):
        dpid = [1]

        def next_dpid():
            val = '{:016x}'.format(dpid[0])
            dpid[0] += 1
            return val

        # ISP LAYER
        isp1 = self.addSwitch('isp1', dpid=next_dpid())
        isp2 = self.addSwitch('isp2', dpid=next_dpid())

        # LOAD BALANCER LAYER
        lb1 = self.addSwitch('lb1', dpid=next_dpid())
        lb2 = self.addSwitch('lb2', dpid=next_dpid())
        self.addLink(isp1, lb1)
        self.addLink(isp1, lb2)
        self.addLink(isp2, lb1)
        self.addLink(isp2, lb2)

        # FIREWALL LAYER
        fw1 = self.addSwitch('fw1', dpid=next_dpid())
        fw2 = self.addSwitch('fw2', dpid=next_dpid())
        self.addLink(lb1, fw1)
        self.addLink(lb1, fw2)
        self.addLink(lb2, fw1)
        self.addLink(lb2, fw2)

        # CORE LAYER
        core_dc = self.addSwitch('core_dc', dpid=next_dpid())
        core_dr = self.addSwitch('core_dr', dpid=next_dpid())
        self.addLink(fw1, core_dc)
        self.addLink(fw2, core_dr)
        self.addLink(core_dc, core_dr)

        # CORE SERVICES
        ipam      = self.addHost('ipam')       # IP address management
        cache     = self.addHost('cache')      # content/web cache
        wifi_ctrl = self.addHost('wifi_ctrl')  # wireless LAN controller
        nms       = self.addHost('nms')        # network management
        self.addLink(ipam,      core_dc)
        self.addLink(cache,     core_dc)
        self.addLink(wifi_ctrl, core_dr)
        self.addLink(nms,       core_dr)

        # DISTRIBUTION LAYER — 5 zones, unchanged
        zone_names = [
            'academic',    # dist1
            'dormitory',   # dist2
            'admin',       # dist3
            'datacenter',  # dist4
            'library',     # dist5
        ]
        dist_switches = []
        for i, zone in enumerate(zone_names, start=1):
            dist = self.addSwitch(f'dist{i}', dpid=next_dpid())
            dist_switches.append(dist)
            self.addLink(core_dc, dist)
            self.addLink(core_dr, dist)

        # ACCESS LAYER — 2 hosts per switch (was 4+4, now 1+1)
        # Keeps campus structure intact with far fewer flows
        for i, dist in enumerate(dist_switches, start=1):
            access1 = self.addSwitch(f'access{i}a', dpid=next_dpid())
            access2 = self.addSwitch(f'access{i}b', dpid=next_dpid())
            self.addLink(access1, access2)
            self.addLink(dist, access1)
            self.addLink(dist, access2)

            # 1 host per access switch = 2 per zone = 10 total end hosts
            host1 = self.addHost(f'h{i}1')
            host2 = self.addHost(f'h{i}2')
            self.addLink(host1, access1)
            self.addLink(host2, access2)


# -------------------------------------------------------------------
# Run
# -------------------------------------------------------------------

def run():
    topo = EnterpriseCampusTopo()
    net = Mininet(
        topo=topo,
        controller=lambda name: RemoteController(name, ip='127.0.0.1', port=6653),
        link=TCLink
    )
    net.start()
    print("*** Network Started ***")
    print(f"*** Switches : 23 (isp x2, lb x2, fw x2, core x2, dist x5, access x10)")
    print(f"*** Hosts    : 14 (10 end hosts + 4 core service servers)")
    print(f"*** Flows    : ~10-20 active (target for RL training speed)")
    print("*** Launching traffic in 3s...")
    time.sleep(3)

    t = threading.Thread(
        target=launch_campus_traffic,
        args=(net,),
        kwargs={'duration': 180},
        daemon=True
    )
    t.start()

    CLI(net)
    net.stop()


if __name__ == '__main__':
    run()

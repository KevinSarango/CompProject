"""
campus_topology.py

Enterprise Campus Topology with role-based hosts and realistic
application traffic simulation. Each host is assigned a role
matching its position in the campus network, and traffic is
generated using iperf3 and ping to simulate real applications:

  - Bulk data transfer  → simulates backup jobs, file server traffic
  - Video streaming     → simulates lecture capture / surveillance feeds
  - VoIP               → simulates IP phone calls between buildings
  - Interactive/web    → simulates HTTP browsing, SSH, database queries
  - Background         → simulates ARP, DNS, NTP keep-alives

This directly addresses the classification realism concern:
traffic types and their mixture reflect what an enterprise campus
actually produces, not random synthetic flows.
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
# Each function uses iperf3 to simulate a specific application type.
# These run inside Mininet host processes via host.cmd().
# -------------------------------------------------------------------

def start_iperf_server(host, port):
    """Start a background iperf3 server on the given host and port."""
    host.cmd(f'iperf3 -s -p {port} -D --logfile /tmp/iperf_{host.name}_{port}.log')
    time.sleep(0.3)


def gen_bulk_transfer(src, dst, port=5101, duration=60):
    """
    Simulates: file server backup, NFS mount, software deployment.
    High throughput TCP, large packets near MTU.
    """
    start_iperf_server(dst, port)
    src.cmd(f'iperf3 -c {dst.IP()} -p {port} -t {duration} -b 100M '
            f'--logfile /tmp/bulk_{src.name}.log &')


def gen_video_stream(src, dst, port=5102, duration=60):
    """
    Simulates: lecture capture upload, IP camera feed, video conf.
    Steady UDP at ~5Mbps, medium packet size (~1000 bytes).
    """
    start_iperf_server(dst, port)
    src.cmd(f'iperf3 -c {dst.IP()} -p {port} -u -b 5M -l 1000 -t {duration} '
            f'--logfile /tmp/video_{src.name}.log &')


def gen_voip(src, dst, port=5103, duration=60):
    """
    Simulates: IP phone call (G.711 codec ~64kbps, ~50pps).
    Tiny UDP packets (~200 bytes) at very steady rate.
    """
    start_iperf_server(dst, port)
    src.cmd(f'iperf3 -c {dst.IP()} -p {port} -u -b 64k -l 200 -t {duration} '
            f'--logfile /tmp/voip_{src.name}.log &')


def gen_interactive(src, dst, port=5104, duration=60):
    """
    Simulates: HTTP/HTTPS browsing, SSH sessions, database queries.
    Bursty low-bandwidth TCP, variable packet sizes.
    """
    start_iperf_server(dst, port)
    src.cmd(f'iperf3 -c {dst.IP()} -p {port} -b 1M -t {duration} '
            f'--logfile /tmp/interactive_{src.name}.log &')


def gen_background(src, dst, count=30, interval=0.5):
    """
    Simulates: DNS lookups, ARP, NTP, SNMP polling.
    Low-rate ICMP ping traffic.
    """
    src.cmd(f'ping -c {count} -i {interval} {dst.IP()} '
            f'> /tmp/bg_{src.name}.log 2>&1 &')


# -------------------------------------------------------------------
# Role-based traffic scenario
# Maps each building/zone to realistic application mix
# -------------------------------------------------------------------

def launch_campus_traffic(net, duration=60):
    """
    Launches a realistic enterprise campus traffic mix.

    Zone assignment (matches topology dist1-dist5):
      dist1 zone -> Academic building  : interactive + video
      dist2 zone -> Dormitory          : bulk + video + background
      dist3 zone -> Admin building     : interactive + voip
      dist4 zone -> Data center access : bulk + background
      dist5 zone -> Library / WiFi     : interactive + background

    This mixture reflects real campus traffic:
    - Academic:   students streaming lecture, doing web research
    - Dorm:       Netflix-like video, large file downloads
    - Admin:      VoIP calls, email/web (interactive)
    - DC access:  server backup/replication (bulk)
    - Library:    light web browsing, background WiFi probes
    """

    hosts = {h.name: h for h in net.hosts}

    def get(name):
        return hosts.get(name)

    print("\n[TRAFFIC] Launching role-based campus traffic simulation...")
    print(f"[TRAFFIC] Duration: {duration}s per flow\n")

    scenarios = [
        # (src_name,   dst_name,    traffic_type,   label)
        # --- dist1: Academic building ---
        ('h11', 'h12', 'video',       'Academic: lecture capture'),
        ('h13', 'h14', 'interactive', 'Academic: web/SSH'),

        # --- dist2: Dormitory ---
        ('h21', 'h22', 'video',       'Dorm: video stream'),
        ('h23', 'h24', 'bulk',        'Dorm: large file download'),
        ('h25', 'h26', 'background',  'Dorm: background/WiFi probe'),

        # --- dist3: Admin building ---
        ('h31', 'h32', 'voip',        'Admin: IP phone call'),
        ('h33', 'h34', 'interactive', 'Admin: email/web'),
        ('h35', 'h36', 'voip',        'Admin: IP phone call 2'),

        # --- dist4: Data center access ---
        ('h41', 'h42', 'bulk',        'DC: server backup'),
        ('h43', 'h44', 'bulk',        'DC: replication traffic'),
        ('h45', 'h46', 'background',  'DC: health check ping'),

        # --- dist5: Library / WiFi zone ---
        ('h51', 'h52', 'interactive', 'Library: web browsing'),
        ('h53', 'h54', 'background',  'Library: background probe'),

        # --- Cross-zone: core services ---
        ('h11', 'cache',    'interactive', 'Cache server request'),
        ('h41', 'ipam',     'background',  'IPAM health check'),
        ('h31', 'wifi_ctrl','background',  'WiFi controller poll'),
        ('h21', 'nms',      'background',  'NMS SNMP poll'),
    ]

    port = 5200
    for src_name, dst_name, ttype, label in scenarios:
        src = get(src_name)
        dst = get(dst_name)
        if src is None or dst is None:
            print(f"[TRAFFIC] Skipping {label} -- host not found")
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

    print(f"\n[TRAFFIC] All flows started. Runs for ~{duration}s.")
    print("[TRAFFIC] Ryu controller will classify flows and apply QoS.")
    print("[TRAFFIC] Check metrics_log.csv for performance data.\n")


# -------------------------------------------------------------------
# Topology
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

        # CORE SERVICES -- named to reflect real campus server roles
        ipam      = self.addHost('ipam')       # IP address management server
        cache     = self.addHost('cache')      # content/web cache server
        wifi_ctrl = self.addHost('wifi_ctrl')  # wireless LAN controller
        nms       = self.addHost('nms')        # network management / SNMP collector
        self.addLink(ipam,      core_dc)
        self.addLink(cache,     core_dc)
        self.addLink(wifi_ctrl, core_dr)
        self.addLink(nms,       core_dr)

        # DISTRIBUTION LAYER -- each dist switch represents a campus zone
        zone_names = [
            'academic',   # dist1
            'dormitory',  # dist2
            'admin',      # dist3
            'datacenter', # dist4
            'library',    # dist5
        ]
        dist_switches = []
        for i, zone in enumerate(zone_names, start=1):
            dist = self.addSwitch(f'dist{i}', dpid=next_dpid())
            dist_switches.append(dist)
            self.addLink(core_dc, dist)
            self.addLink(core_dr, dist)

        # ACCESS LAYER + END HOSTS
        for i, dist in enumerate(dist_switches, start=1):
            access1 = self.addSwitch(f'access{i}a', dpid=next_dpid())
            access2 = self.addSwitch(f'access{i}b', dpid=next_dpid())
            self.addLink(access1, access2)
            self.addLink(dist, access1)
            self.addLink(dist, access2)

            for j in range(1, 5):
                host = self.addHost(f'h{i}{j}')
                self.addLink(host, access1)
            for j in range(5, 9):
                host = self.addHost(f'h{i}{j}')
                self.addLink(host, access2)

def run():
    topo = EnterpriseCampusTopo()
    net = Mininet(
        topo=topo,
        controller=lambda name: RemoteController(name, ip='127.0.0.1', port=6653),
        link=TCLink
    )
    net.start()
    print("*** Network Started ***")
    print("*** Launching campus traffic simulation in 3s...")
    time.sleep(3)

    t = threading.Thread(
        target=launch_campus_traffic,
        args=(net,),
        kwargs={'duration': 120},
        daemon=True
    )
    t.start()

    CLI(net)
    net.stop()


if __name__ == '__main__':
    run()
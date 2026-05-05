"""
traffic_generator.py

Run inside Mininet CLI or via net.hosts to generate realistic
multi-class traffic between hosts.

Usage from Mininet CLI:
  mininet> h11 python3 traffic_generator.py server &
  mininet> h12 python3 traffic_generator.py client h11

Or launch all traffic automatically by calling generate_all_traffic(net)
from your campus_togology.py run() function.
"""

import subprocess
import random
import time
import sys
import threading


# -------------------------------------------------------------------
# Individual traffic generators using standard Linux tools
# -------------------------------------------------------------------

def gen_bulk_transfer(src_host, dst_ip, port=5201, duration=30):
    """iperf3 TCP — simulates file transfer / backup (bulk_data)"""
    # Server side
    server_cmd = f"iperf3 -s -p {port} -1"
    # Client side
    client_cmd = f"iperf3 -c {dst_ip} -p {port} -t {duration} -b 100M"
    return server_cmd, client_cmd


def gen_video_stream(src_host, dst_ip, port=5002, duration=60):
    """iperf3 UDP at fixed bitrate — simulates video stream"""
    server_cmd = f"iperf3 -s -p {port} -1"
    client_cmd = f"iperf3 -c {dst_ip} -p {port} -u -b 5M -t {duration}"
    return server_cmd, client_cmd


def gen_voip(src_host, dst_ip, port=5004, duration=60):
    """iperf3 UDP tiny packets at 50pps — simulates VoIP/RTP"""
    server_cmd = f"iperf3 -s -p {port} -1"
    client_cmd = f"iperf3 -c {dst_ip} -p {port} -u -b 64k -l 200 -t {duration}"
    return server_cmd, client_cmd


def gen_interactive(src_host, dst_ip, port=5006, duration=30):
    """iperf3 TCP low bandwidth bursty — simulates SSH/web browsing"""
    server_cmd = f"iperf3 -s -p {port} -1"
    client_cmd = f"iperf3 -c {dst_ip} -p {port} -b 1M -t {duration} --connect-timeout 2000"
    return server_cmd, client_cmd


def gen_background(dst_ip):
    """ping — simulates background keep-alive / ARP traffic"""
    return f"ping -c 20 -i 0.5 {dst_ip}"


# -------------------------------------------------------------------
# Mininet-integrated launcher
# Call this from your run() function after net.start()
# -------------------------------------------------------------------
def generate_all_traffic(net, duration=60):
    """
    Launches mixed traffic across the topology to create diverse
    flow data for the ML classifier to observe.
    """
    hosts = net.hosts
    if len(hosts) < 4:
        print("[TRAFFIC] Not enough hosts to generate traffic")
        return

    print(f"[TRAFFIC] Launching {duration}s mixed traffic scenario...")
    threads = []

    def run_pair(src, dst, traffic_type, port):
        dst_ip = dst.IP()
        if traffic_type == 'bulk':
            _, client_cmd = gen_bulk_transfer(src, dst_ip, port, duration)
            dst.cmd(f"iperf3 -s -p {port} -D")
            time.sleep(0.5)
            src.cmd(client_cmd)
        elif traffic_type == 'video':
            _, client_cmd = gen_video_stream(src, dst_ip, port, duration)
            dst.cmd(f"iperf3 -s -p {port} -D")
            time.sleep(0.5)
            src.cmd(client_cmd)
        elif traffic_type == 'voip':
            _, client_cmd = gen_voip(src, dst_ip, port, duration)
            dst.cmd(f"iperf3 -s -p {port} -D")
            time.sleep(0.5)
            src.cmd(client_cmd)
        elif traffic_type == 'interactive':
            _, client_cmd = gen_interactive(src, dst_ip, port, duration)
            dst.cmd(f"iperf3 -s -p {port} -D")
            time.sleep(0.5)
            src.cmd(client_cmd)
        elif traffic_type == 'background':
            src.cmd(gen_background(dst_ip))

    # Assign traffic types across host pairs
    scenarios = [
        (hosts[0],  hosts[1],  'bulk',        5201),
        (hosts[2],  hosts[3],  'video',        5202),
        (hosts[4],  hosts[5],  'voip',         5203),
        (hosts[6],  hosts[7],  'interactive',  5204),
        (hosts[8],  hosts[9],  'background',   5205),
        (hosts[10], hosts[11], 'bulk',         5206),
        (hosts[12], hosts[13], 'video',        5207),
        (hosts[14], hosts[15], 'voip',         5208),
    ]

    for src, dst, ttype, port in scenarios:
        t = threading.Thread(
            target=run_pair,
            args=(src, dst, ttype, port),
            daemon=True
        )
        threads.append(t)
        t.start()
        time.sleep(0.2)   # stagger starts

    print(f"[TRAFFIC] All flows launched. Running for {duration}s...")
    time.sleep(duration + 5)
    print("[TRAFFIC] Traffic generation complete.")


# -------------------------------------------------------------------
# Scalability test — ramp up number of concurrent flows
# to evaluate controller overhead vs. traffic volume
# -------------------------------------------------------------------
def scalability_test(net, max_pairs=20, step=5, duration_each=30):
    """
    Incrementally increase concurrent flows and record how the
    controller (Ryu) handles the increased classification workload.
    Results are read from metrics_log.csv after each phase.
    """
    hosts = net.hosts
    print("\n[SCALE TEST] Starting scalability evaluation...")

    for n_pairs in range(step, min(max_pairs + 1, len(hosts) // 2 + 1), step):
        print(f"\n[SCALE TEST] Phase: {n_pairs} concurrent flow pairs")
        threads = []

        for i in range(n_pairs):
            src = hosts[i * 2]
            dst = hosts[i * 2 + 1]
            port = 6000 + i
            dst.cmd(f"iperf3 -s -p {port} -D")
            time.sleep(0.1)

            t = threading.Thread(
                target=lambda s=src, d=dst, p=port:
                    s.cmd(f"iperf3 -c {d.IP()} -p {p} -t {duration_each} -b 10M"),
                daemon=True
            )
            threads.append(t)
            t.start()

        time.sleep(duration_each + 3)
        print(f"[SCALE TEST] Phase {n_pairs} pairs complete. "
              f"Check metrics_log.csv for overhead data.")

    print("\n[SCALE TEST] Done. Analyze metrics_log.csv for scaling results.")

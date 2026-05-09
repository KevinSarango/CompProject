import csv
import os
import re
import time


TRAFFIC_PAIRS = [
    ("h1", "h5"),
    ("h1", "h6"),
    ("h2", "h7"),
    ("h3", "h8"),
    ("h4", "h5"),
    ("h2", "h6"),
]


def host_ip(host_name):
    number = int(host_name.replace("h", ""))
    return f"10.0.0.{number}"


def parse_ping_output(output):
    packet_loss = 100.0
    latency = 0.0

    loss_match = re.search(r"(\d+(?:\.\d+)?)% packet loss", output)
    if loss_match:
        packet_loss = float(loss_match.group(1))

    rtt_match = re.search(r"rtt min/avg/max/(?:mdev|stddev) = [\d.]+/([\d.]+)/", output)
    if rtt_match:
        latency = float(rtt_match.group(1))

    return latency, packet_loss


def parse_iperf_output(output):
    throughput = 0.0

    lines = output.strip().splitlines()

    for line in reversed(lines):
        if "bits/sec" not in line:
            continue

        match = re.search(r"([\d.]+)\s+([KMG])bits/sec", line)
        if not match:
            continue

        value = float(match.group(1))
        unit = match.group(2)

        if unit == "K":
            throughput = value / 1000.0
        elif unit == "M":
            throughput = value
        elif unit == "G":
            throughput = value * 1000.0

        break

    return throughput


def run_iperf_pair(net, src_name, dst_name):
    src = net.get(src_name)
    dst = net.get(dst_name)

    print(f"[IPERF] {src_name} -> {dst_name}")

    dst.cmd("iperf -s -p 5001 > /tmp/iperf_server.log 2>&1 &")
    time.sleep(1)

    output = src.cmd(f"iperf -c {host_ip(dst_name)} -p 5001 -t 5")
    dst.cmd("pkill -f 'iperf -s'")

    throughput = parse_iperf_output(output)
    return throughput


def run_ping_pair(net, src_name, dst_name):
    src = net.get(src_name)

    print(f"[PING] {src_name} -> {dst_name}")

    output = src.cmd(f"ping -c 20 {host_ip(dst_name)}")
    latency, packet_loss = parse_ping_output(output)

    return latency, packet_loss


def run_automated_tests(net, policy_name):
    os.makedirs("data", exist_ok=True)

    output_file = f"data/{policy_name.lower()}_traffic_metrics.csv"

    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "policy",
            "src",
            "dst",
            "throughput_mbps",
            "latency_ms",
            "packet_loss_percent",
        ])

        print()
        print("====================================")
        print(f"Running automated traffic tests: {policy_name}")
        print("====================================")

        net.pingAll()

        for src, dst in TRAFFIC_PAIRS:
            latency, packet_loss = run_ping_pair(net, src, dst)
            throughput = run_iperf_pair(net, src, dst)

            writer.writerow([
                policy_name,
                src,
                dst,
                throughput,
                latency,
                packet_loss,
            ])

        print()
        print(f"Saved traffic metrics to {output_file}")
        print("====================================")
        print()

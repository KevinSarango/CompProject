import csv
import os
import re
import time


DEMAND_FILE = "data/trafpy_demands.csv"


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

    for line in reversed(output.strip().splitlines()):
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


def load_demands():
    if not os.path.exists(DEMAND_FILE):
        raise FileNotFoundError(
            f"{DEMAND_FILE} not found. Run python3 generate_trafpy_demands.py first."
        )

    demands = []

    with open(DEMAND_FILE, "r") as f:
        reader = csv.DictReader(f)

        for row in reader:
            demands.append({
                "flow_id": int(row["flow_id"]),
                "src": row["src"],
                "dst": row["dst"],
                "start_time": float(row["start_time"]),
                "size_kb": float(row["size_kb"]),
            })

    return demands


def clean_old_logs(net):
    for host in net.hosts:
        host.cmd("rm -f /tmp/iperf_client_*.log /tmp/iperf_server_*.log /tmp/ping_*.log")
        host.cmd("pkill -f iperf 2>/dev/null")
        host.cmd("pkill -f ping 2>/dev/null")


def start_iperf_server(net, dst_name, flow_id, port):
    dst = net.get(dst_name)
    dst.cmd(f"iperf -s -p {port} > /tmp/iperf_server_{flow_id}.log 2>&1 &")


def start_iperf_client(net, src_name, dst_name, flow_id, port, size_kb):
    src = net.get(src_name)
    dst_ip = host_ip(dst_name)

    size_bytes = int(size_kb * 1024)

    src.cmd(
        f"iperf -c {dst_ip} -p {port} -n {size_bytes} "
        f"> /tmp/iperf_client_{flow_id}.log 2>&1 &"
    )


def start_ping_probe(net, src_name, dst_name, flow_id):
    src = net.get(src_name)
    dst_ip = host_ip(dst_name)

    src.cmd(
        f"ping -c 5 {dst_ip} "
        f"> /tmp/ping_{flow_id}.log 2>&1 &"
    )


def read_host_file(net, host_name, path):
    host = net.get(host_name)
    return host.cmd(f"cat {path} 2>/dev/null")


def wait_for_traffic_to_finish(net, timeout=60):
    start = time.time()

    while time.time() - start < timeout:
        still_running = False

        for host in net.hosts:
            output = host.cmd("pgrep -f 'iperf -c|ping -c 5' 2>/dev/null")
            if output.strip():
                still_running = True
                break

        if not still_running:
            return

        time.sleep(0.5)

    print("[WARN] Traffic timeout reached. Continuing anyway.")


def run_automated_tests(net, policy_name):
    os.makedirs("data", exist_ok=True)

    demands = load_demands()
    output_file = f"data/{policy_name.lower()}_traffic_metrics.csv"

    clean_old_logs(net)

    print()
    print("====================================")
    print(f"Running TrafPy-style traffic: {policy_name}")
    print("====================================")

    net.pingAll()

    # Start all iperf servers first.
    for demand in demands:
        flow_id = demand["flow_id"]
        dst = demand["dst"]
        port = 5001 + flow_id
        start_iperf_server(net, dst, flow_id, port)

    time.sleep(1)

    experiment_start = time.time()

    # Launch clients and ping probes according to start_time.
    for demand in demands:
        now = time.time() - experiment_start
        wait_time = demand["start_time"] - now

        if wait_time > 0:
            time.sleep(wait_time)

        flow_id = demand["flow_id"]
        src = demand["src"]
        dst = demand["dst"]
        size_kb = demand["size_kb"]
        port = 5001 + flow_id

        print(
            f"[FLOW {flow_id}] {src} -> {dst}, "
            f"size={size_kb} KB, start={demand['start_time']}s"
        )

        start_ping_probe(net, src, dst, flow_id)
        start_iperf_client(net, src, dst, flow_id, port, size_kb)

    print("[INFO] Waiting for background traffic to finish...")
    wait_for_traffic_to_finish(net, timeout=60)

    # Stop any remaining servers.
    for host in net.hosts:
        host.cmd("pkill -f 'iperf -s' 2>/dev/null")

    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "policy",
            "flow_id",
            "src",
            "dst",
            "size_kb",
            "throughput_mbps",
            "latency_ms",
            "packet_loss_percent",
        ])

        for demand in demands:
            flow_id = demand["flow_id"]
            src = demand["src"]
            dst = demand["dst"]

            iperf_output = read_host_file(
                net,
                src,
                f"/tmp/iperf_client_{flow_id}.log",
            )

            ping_output = read_host_file(
                net,
                src,
                f"/tmp/ping_{flow_id}.log",
            )

            throughput = parse_iperf_output(iperf_output)
            latency, packet_loss = parse_ping_output(ping_output)

            writer.writerow([
                policy_name,
                flow_id,
                src,
                dst,
                demand["size_kb"],
                throughput,
                latency,
                packet_loss,
            ])

    print()
    print(f"Saved traffic metrics to {output_file}")
    print("====================================")
    print()

import csv
import os
import re
import time

from config import (
    FIFO_TRAFFIC_FILE,
    PINGALL_ATTEMPTS,
    PINGALL_RETRY_WAIT_SECONDS,
    RL_TRAFFIC_FILE,
)


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


def cleanup_logs(net):
    for host in net.hosts:
        host.cmd("rm -f /tmp/iperf_client_*.log /tmp/iperf_server_*.log /tmp/ping_*.log")


def run_pingall_with_retries(net):
    print()
    print("====================================")
    print(" Checking connectivity with pingall")
    print("====================================")

    for attempt in range(1, PINGALL_ATTEMPTS + 1):
        print(f"[PINGALL] Attempt {attempt}/{PINGALL_ATTEMPTS}")

        loss = net.pingAll()

        try:
            loss_value = float(loss)
        except (TypeError, ValueError):
            loss_value = 100.0

        if loss_value == 0.0:
            print("[PINGALL] Success: 0% packet loss")
            return True

        print(f"[PINGALL] Failed attempt {attempt}: {loss_value}% packet loss")

        if attempt < PINGALL_ATTEMPTS:
            print(f"[PINGALL] Retrying in {PINGALL_RETRY_WAIT_SECONDS} seconds...")
            time.sleep(PINGALL_RETRY_WAIT_SECONDS)

    print("[PINGALL] Failed after all attempts. Exiting automated traffic test.")
    return False


def get_output_file(policy_name):
    if policy_name.upper() == "FIFO":
        return FIFO_TRAFFIC_FILE

    return RL_TRAFFIC_FILE


def run_automated_tests(net, policy_name):
    os.makedirs("data", exist_ok=True)

    demands = load_demands()
    output_file = get_output_file(policy_name)

    print()
    print("====================================")
    print(f"Running TrafPy-style traffic: {policy_name}")
    print("====================================")

    cleanup_logs(net)

    if not run_pingall_with_retries(net):
        return False

    for demand in demands:
        flow_id = demand["flow_id"]
        dst = net.get(demand["dst"])
        port = 5001 + flow_id

        dst.cmd(
            f"iperf -s -p {port} "
            f"> /tmp/iperf_server_{flow_id}.log 2>&1 &"
        )

    time.sleep(1)

    for demand in demands:
        flow_id = demand["flow_id"]
        src = net.get(demand["src"])
        dst_ip = host_ip(demand["dst"])
        port = 5001 + flow_id
        start_time = demand["start_time"]
        size_bytes = int(demand["size_kb"] * 1024)

        print(
            f"[FLOW {flow_id}] {demand['src']} -> {demand['dst']} "
            f"start={start_time}s size={demand['size_kb']}KB port={port}"
        )

        src.cmd(
            f"sh -c 'sleep {start_time}; "
            f"iperf -c {dst_ip} -p {port} -n {size_bytes} "
            f"> /tmp/iperf_client_{flow_id}.log 2>&1' &"
        )

        src.cmd(
            f"sh -c 'sleep {start_time}; "
            f"ping -c 5 {dst_ip} "
            f"> /tmp/ping_{flow_id}.log 2>&1' &"
        )

    max_start = max(d["start_time"] for d in demands)
    wait_time = max_start + 45

    print("[INFO] Waiting for traffic to finish...")
    time.sleep(wait_time)

    for host in net.hosts:
        host.cmd("pkill -f 'iperf -s'")

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
            src = net.get(demand["src"])

            iperf_output = src.cmd(f"cat /tmp/iperf_client_{flow_id}.log 2>/dev/null")
            ping_output = src.cmd(f"cat /tmp/ping_{flow_id}.log 2>/dev/null")

            throughput = parse_iperf_output(iperf_output)
            latency, packet_loss = parse_ping_output(ping_output)

            writer.writerow([
                policy_name,
                flow_id,
                demand["src"],
                demand["dst"],
                demand["size_kb"],
                throughput,
                latency,
                packet_loss,
            ])

    print()
    print(f"Saved traffic metrics to {output_file}")
    print("====================================")
    print()

    return True

import csv
import os
import re
import time

from config import (
    EVAL_BASE_SEED,
    EVAL_NUM_FLOWS,
    EVAL_NUM_RUNS,
    EVAL_SEEDS_FILE,
    FIFO_TRAFFIC_FILE,
    PINGALL_ATTEMPTS,
    PINGALL_RETRY_WAIT_SECONDS,
    PLOT_PREFIX,
    RL_TRAFFIC_FILE,
)
from generate_trafpy_demands import generate_demands, save_demands


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


def save_eval_seeds(seeds):
    os.makedirs("data", exist_ok=True)

    with open(EVAL_SEEDS_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["eval_run", "eval_seed", "num_flows"])

        for eval_run, seed in enumerate(seeds):
            writer.writerow([eval_run, seed, EVAL_NUM_FLOWS])


def start_iperf_servers(net, demands):
    for demand in demands:
        flow_id = demand["flow_id"]
        dst = net.get(demand["dst"])
        port = 5001 + flow_id

        dst.cmd(
            f"iperf -s -p {port} "
            f"> /tmp/iperf_server_{flow_id}.log 2>&1 &"
        )


def start_flow_clients(net, demands, eval_run, eval_seed):
    for demand in demands:
        flow_id = demand["flow_id"]
        src = net.get(demand["src"])
        dst_ip = host_ip(demand["dst"])
        port = 5001 + flow_id
        start_time = demand["start_time"]
        size_bytes = int(demand["size_kb"] * 1024)

        print(
            f"[EVAL {eval_run} seed={eval_seed}] "
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


def collect_run_metrics(net, writer, policy_name, demands, eval_run, eval_seed):
    for demand in demands:
        flow_id = demand["flow_id"]
        src = net.get(demand["src"])

        iperf_output = src.cmd(f"cat /tmp/iperf_client_{flow_id}.log 2>/dev/null")
        ping_output = src.cmd(f"cat /tmp/ping_{flow_id}.log 2>/dev/null")

        throughput = parse_iperf_output(iperf_output)
        latency, packet_loss = parse_ping_output(ping_output)

        writer.writerow([
            policy_name,
            eval_run,
            eval_seed,
            flow_id,
            demand["src"],
            demand["dst"],
            demand["size_kb"],
            throughput,
            latency,
            packet_loss,
        ])


def run_single_eval_trace(net, policy_name, writer, eval_run, eval_seed):
    demands = generate_demands(num_flows=EVAL_NUM_FLOWS, seed=eval_seed)

    # Save the active trace for Ryu. The RL controller reloads this file when
    # its mtime changes and maps TCP port 5001 + flow_id back to size_kb.
    save_demands(demands, output_file=DEMAND_FILE, verbose=False)

    archive_file = f"data/{PLOT_PREFIX}_eval_demands_seed_{eval_seed}.csv"
    save_demands(demands, output_file=archive_file, verbose=False)

    print()
    print("====================================")
    print(f"Running {policy_name} eval_run={eval_run} seed={eval_seed}")
    print("====================================")

    cleanup_logs(net)

    start_iperf_servers(net, demands)
    time.sleep(1)

    start_flow_clients(net, demands, eval_run, eval_seed)

    max_start = max(d["start_time"] for d in demands) if demands else 0.0
    wait_time = max_start + 45

    print("[INFO] Waiting for traffic to finish...")
    time.sleep(wait_time)

    for host in net.hosts:
        host.cmd("pkill -f 'iperf -s'")

    collect_run_metrics(net, writer, policy_name, demands, eval_run, eval_seed)


def run_automated_tests(net, policy_name):
    os.makedirs("data", exist_ok=True)

    output_file = get_output_file(policy_name)
    eval_seeds = [EVAL_BASE_SEED + index for index in range(EVAL_NUM_RUNS)]
    save_eval_seeds(eval_seeds)

    print()
    print("====================================")
    print(f"Running multi-seed TrafPy-style traffic: {policy_name}")
    print(f"eval_runs={EVAL_NUM_RUNS} flows_per_run={EVAL_NUM_FLOWS}")
    print("====================================")

    cleanup_logs(net)

    if not run_pingall_with_retries(net):
        return False

    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "policy",
            "eval_run",
            "eval_seed",
            "flow_id",
            "src",
            "dst",
            "size_kb",
            "throughput_mbps",
            "latency_ms",
            "packet_loss_percent",
        ])

        for eval_run, eval_seed in enumerate(eval_seeds):
            run_single_eval_trace(net, policy_name, writer, eval_run, eval_seed)

    print()
    print(f"Saved traffic metrics to {output_file}")
    print(f"Saved evaluation seed manifest to {EVAL_SEEDS_FILE}")
    print("====================================")
    print()

    return True

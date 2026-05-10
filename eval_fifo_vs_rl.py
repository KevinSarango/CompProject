import csv
import os
from collections import Counter

import matplotlib.pyplot as plt

from config import (
    FIFO_METRICS_FILE,
    FIFO_TRAFFIC_FILE,
    PATHS,
    PLOTS_DIR,
    PLOT_PREFIX,
    RL_METRICS_FILE,
    RL_TRAFFIC_FILE,
    TOPO_MODE,
)


def ensure_dirs():
    os.makedirs(PLOTS_DIR, exist_ok=True)


def read_controller_decisions(path):
    if not os.path.exists(path):
        print(f"[WARN] Missing controller decision file: {path}")
        return []

    rows = []

    with open(path, "r") as f:
        reader = csv.DictReader(f)

        for row in reader:
            proto = row.get("proto", "").lower()
            path_name = row.get("path", "")

            if proto != "tcp":
                continue

            if path_name not in PATHS:
                continue

            rows.append(row)

    return rows


def summarize_path_usage(decision_rows):
    paths = [row["path"] for row in decision_rows]
    counts = Counter(paths)
    total = sum(counts.values())

    summary = {"total_flows": total}

    for path in PATHS:
        count = counts[path]
        percent = round((count / total) * 100, 2) if total else 0.0
        summary[path] = count
        summary[f"{path}_percent"] = percent

    return summary


def read_traffic_metrics(path):
    if not os.path.exists(path):
        print(f"[WARN] Missing traffic metrics file: {path}")
        return []

    rows = []

    with open(path, "r") as f:
        reader = csv.DictReader(f)

        for row in reader:
            try:
                rows.append({
                    "policy": row["policy"],
                    "flow_id": int(row["flow_id"]),
                    "src": row["src"],
                    "dst": row["dst"],
                    "size_kb": float(row["size_kb"]),
                    "throughput_mbps": float(row["throughput_mbps"]),
                    "latency_ms": float(row["latency_ms"]),
                    "packet_loss_percent": float(row["packet_loss_percent"]),
                })
            except (KeyError, ValueError) as e:
                print(f"[WARN] Skipping bad traffic row in {path}: {row} ({e})")

    rows.sort(key=lambda r: r["flow_id"])
    return rows


def average_metric(rows, metric):
    if not rows:
        return 0.0

    return sum(row[metric] for row in rows) / len(rows)


def summarize_traffic(rows):
    return {
        "total_tests": len(rows),
        "average_throughput_mbps": average_metric(rows, "throughput_mbps"),
        "average_latency_ms": average_metric(rows, "latency_ms"),
        "average_packet_loss_percent": average_metric(rows, "packet_loss_percent"),
    }


def print_path_summary(name, summary):
    print()
    print(f"{name} Controller TCP Path Decisions ({TOPO_MODE})")
    print("-" * 50)
    print(f"TCP decisions: {summary['total_flows']}")

    for path in PATHS:
        print(f"{path}: {summary[path]} ({summary[f'{path}_percent']}%)")


def print_traffic_summary(name, summary):
    print()
    print(f"{name} Traffic Metrics ({TOPO_MODE})")
    print("-" * 50)
    print(f"Traffic tests:        {summary['total_tests']}")
    print(f"Average throughput:   {summary['average_throughput_mbps']:.3f} Mbps")
    print(f"Average latency:      {summary['average_latency_ms']:.3f} ms")
    print(f"Average packet loss:  {summary['average_packet_loss_percent']:.3f}%")


def validate_counts(name, decision_rows, traffic_rows):
    decision_count = len(decision_rows)
    traffic_count = len(traffic_rows)

    print()
    print(f"{name} Validation")
    print("-" * 50)
    print(f"TCP controller decisions: {decision_count}")
    print(f"Traffic metric rows:      {traffic_count}")

    if traffic_count == 0:
        print("[WARN] No traffic rows found.")
        return

    ratio = decision_count / traffic_count

    if ratio < 0.8:
        print("[WARN] Controller decisions are much lower than traffic rows.")
    elif ratio > 1.5:
        print("[WARN] Controller decisions are much higher than traffic rows.")
    else:
        print("[OK] Controller decisions roughly match traffic tests.")


def plot_path_usage(fifo_summary, rl_summary):
    fifo_values = [fifo_summary[path] for path in PATHS]
    rl_values = [rl_summary[path] for path in PATHS]

    x = range(len(PATHS))

    plt.figure()
    plt.bar([i - 0.2 for i in x], fifo_values, width=0.4, label="FIFO")
    plt.bar([i + 0.2 for i in x], rl_values, width=0.4, label="RL")
    plt.xticks(list(x), PATHS, rotation=15)
    plt.ylabel("TCP Flow Decisions")
    plt.title(f"FIFO vs RL TCP Path Decisions ({TOPO_MODE})")
    plt.legend()

    output = os.path.join(PLOTS_DIR, f"{PLOT_PREFIX}_fifo_vs_rl_path_usage.png")
    plt.savefig(output, bbox_inches="tight")
    plt.close()


def plot_metric_line_graph(fifo_rows, rl_rows, metric, ylabel, title, filename):
    fifo_values = [row[metric] for row in fifo_rows]
    rl_values = [row[metric] for row in rl_rows]

    fifo_x = [row["flow_id"] for row in fifo_rows]
    rl_x = [row["flow_id"] for row in rl_rows]

    plt.figure()

    if fifo_values:
        plt.plot(fifo_x, fifo_values, marker="o", label="FIFO")

    if rl_values:
        plt.plot(rl_x, rl_values, marker="o", label="RL")

    plt.xlabel("TrafPy Flow ID")
    plt.ylabel(ylabel)
    plt.title(f"{title} ({TOPO_MODE})")
    plt.legend()

    output = os.path.join(PLOTS_DIR, f"{PLOT_PREFIX}_{filename}")
    plt.savefig(output, bbox_inches="tight")
    plt.close()


def main():
    ensure_dirs()

    fifo_decisions = read_controller_decisions(FIFO_METRICS_FILE)
    rl_decisions = read_controller_decisions(RL_METRICS_FILE)

    fifo_path_summary = summarize_path_usage(fifo_decisions)
    rl_path_summary = summarize_path_usage(rl_decisions)

    fifo_rows = read_traffic_metrics(FIFO_TRAFFIC_FILE)
    rl_rows = read_traffic_metrics(RL_TRAFFIC_FILE)

    fifo_traffic_summary = summarize_traffic(fifo_rows)
    rl_traffic_summary = summarize_traffic(rl_rows)

    print_path_summary("FIFO", fifo_path_summary)
    print_path_summary("RL", rl_path_summary)

    print_traffic_summary("FIFO", fifo_traffic_summary)
    print_traffic_summary("RL", rl_traffic_summary)

    validate_counts("FIFO", fifo_decisions, fifo_rows)
    validate_counts("RL", rl_decisions, rl_rows)

    plot_path_usage(fifo_path_summary, rl_path_summary)

    plot_metric_line_graph(
        fifo_rows,
        rl_rows,
        metric="throughput_mbps",
        ylabel="Throughput (Mbps)",
        title="FIFO vs RL Throughput",
        filename="fifo_vs_rl_throughput.png",
    )

    plot_metric_line_graph(
        fifo_rows,
        rl_rows,
        metric="latency_ms",
        ylabel="Latency (ms)",
        title="FIFO vs RL Latency",
        filename="fifo_vs_rl_latency.png",
    )

    plot_metric_line_graph(
        fifo_rows,
        rl_rows,
        metric="packet_loss_percent",
        ylabel="Packet Loss (%)",
        title="FIFO vs RL Packet Loss",
        filename="fifo_vs_rl_packet_loss.png",
    )

    print()
    print("Saved plots in data/plots/")
    print(f"- {PLOT_PREFIX}_fifo_vs_rl_path_usage.png")
    print(f"- {PLOT_PREFIX}_fifo_vs_rl_throughput.png")
    print(f"- {PLOT_PREFIX}_fifo_vs_rl_latency.png")
    print(f"- {PLOT_PREFIX}_fifo_vs_rl_packet_loss.png")


if __name__ == "__main__":
    main()

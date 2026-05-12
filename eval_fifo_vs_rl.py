import csv
import os
from collections import Counter, defaultdict
from statistics import mean, median

import matplotlib.pyplot as plt

from config import (
    EVAL_BY_SEED_SUMMARY_FILE,
    EVAL_SUMMARY_FILE,
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
    os.makedirs("data", exist_ok=True)


def percentile(values, pct):
    if not values:
        return 0.0

    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]

    rank = (pct / 100.0) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    fraction = rank - low
    return ordered[low] + (ordered[high] - ordered[low]) * fraction


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

        for index, row in enumerate(reader):
            try:
                rows.append({
                    "policy": row["policy"],
                    "eval_run": int(row.get("eval_run", 0) or 0),
                    "eval_seed": int(row.get("eval_seed", 0) or 0),
                    "flow_id": int(row["flow_id"]),
                    "src": row["src"],
                    "dst": row["dst"],
                    "size_kb": float(row["size_kb"]),
                    "throughput_mbps": float(row["throughput_mbps"]),
                    "latency_ms": float(row["latency_ms"]),
                    "packet_loss_percent": float(row["packet_loss_percent"]),
                    "test_index": index,
                })
            except (KeyError, ValueError) as e:
                print(f"[WARN] Skipping bad traffic row in {path}: {row} ({e})")

    rows.sort(key=lambda r: (r["eval_run"], r["flow_id"]))

    for index, row in enumerate(rows):
        row["test_index"] = index

    return rows


def metric_values(rows, metric):
    return [row[metric] for row in rows]


def summarize_traffic(rows):
    throughput = metric_values(rows, "throughput_mbps")
    latency = metric_values(rows, "latency_ms")
    packet_loss = metric_values(rows, "packet_loss_percent")
    lossy_flows = sum(1 for value in packet_loss if value > 0.0)

    return {
        "total_tests": len(rows),
        "mean_throughput_mbps": mean(throughput) if throughput else 0.0,
        "median_throughput_mbps": median(throughput) if throughput else 0.0,
        "p95_throughput_mbps": percentile(throughput, 95),
        "mean_latency_ms": mean(latency) if latency else 0.0,
        "median_latency_ms": median(latency) if latency else 0.0,
        "p95_latency_ms": percentile(latency, 95),
        "mean_packet_loss_percent": mean(packet_loss) if packet_loss else 0.0,
        "median_packet_loss_percent": median(packet_loss) if packet_loss else 0.0,
        "p95_packet_loss_percent": percentile(packet_loss, 95),
        "lossy_flows": lossy_flows,
        "lossy_flow_percent": (lossy_flows / len(rows)) * 100.0 if rows else 0.0,
    }


def summarize_by_seed(rows):
    grouped = defaultdict(list)

    for row in rows:
        grouped[row["eval_run"]].append(row)

    summaries = {}

    for eval_run, seed_rows in sorted(grouped.items()):
        seed = seed_rows[0]["eval_seed"]
        summaries[(eval_run, seed)] = summarize_traffic(seed_rows)

    return summaries


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
    print(f"Traffic tests:                 {summary['total_tests']}")
    print(f"Mean throughput:               {summary['mean_throughput_mbps']:.3f} Mbps")
    print(f"Median throughput:             {summary['median_throughput_mbps']:.3f} Mbps")
    print(f"95th percentile throughput:    {summary['p95_throughput_mbps']:.3f} Mbps")
    print(f"Mean latency:                  {summary['mean_latency_ms']:.3f} ms")
    print(f"Median latency:                {summary['median_latency_ms']:.3f} ms")
    print(f"95th percentile latency:       {summary['p95_latency_ms']:.3f} ms")
    print(f"Mean packet loss:              {summary['mean_packet_loss_percent']:.3f}%")
    print(f"Median packet loss:            {summary['median_packet_loss_percent']:.3f}%")
    print(f"95th percentile packet loss:   {summary['p95_packet_loss_percent']:.3f}%")
    print(f"Lossy flows:                   {summary['lossy_flows']} ({summary['lossy_flow_percent']:.2f}%)")


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

    fifo_x = [row["test_index"] for row in fifo_rows]
    rl_x = [row["test_index"] for row in rl_rows]

    plt.figure()

    if fifo_values:
        plt.plot(fifo_x, fifo_values, marker="o", label="FIFO")

    if rl_values:
        plt.plot(rl_x, rl_values, marker="o", label="RL")

    plt.xlabel("Evaluation sample index (all seeds combined)")
    plt.ylabel(ylabel)
    plt.title(f"{title} ({TOPO_MODE})")
    plt.legend()

    output = os.path.join(PLOTS_DIR, f"{PLOT_PREFIX}_{filename}")
    plt.savefig(output, bbox_inches="tight")
    plt.close()


def plot_summary_bars(fifo_summary, rl_summary):
    plots = [
        (
            "Latency Summary",
            ["mean_latency_ms", "median_latency_ms", "p95_latency_ms"],
            ["Mean", "Median", "P95"],
            "Latency (ms)",
            "latency_summary.png",
        ),
        (
            "Throughput Summary",
            ["mean_throughput_mbps", "median_throughput_mbps", "p95_throughput_mbps"],
            ["Mean", "Median", "P95"],
            "Throughput (Mbps)",
            "throughput_summary.png",
        ),
        (
            "Packet Loss Summary",
            ["mean_packet_loss_percent", "median_packet_loss_percent", "p95_packet_loss_percent"],
            ["Mean", "Median", "P95"],
            "Packet Loss (%)",
            "packet_loss_summary.png",
        ),
        (
            "Lossy Flow Summary",
            ["lossy_flow_percent"],
            ["Lossy flows"],
            "Flows with packet loss (%)",
            "lossy_flow_summary.png",
        ),
    ]

    for title, metrics, labels, ylabel, filename in plots:
        x = range(len(metrics))
        fifo_values = [fifo_summary[m] for m in metrics]
        rl_values = [rl_summary[m] for m in metrics]

        plt.figure()
        plt.bar([i - 0.2 for i in x], fifo_values, width=0.4, label="FIFO")
        plt.bar([i + 0.2 for i in x], rl_values, width=0.4, label="RL")
        plt.xticks(list(x), labels)
        plt.ylabel(ylabel)
        plt.title(f"{title} ({TOPO_MODE})")
        plt.legend()

        output = os.path.join(PLOTS_DIR, f"{PLOT_PREFIX}_{filename}")
        plt.savefig(output, bbox_inches="tight")
        plt.close()


def write_summary_csv(fifo_summary, rl_summary):
    fieldnames = [
        "policy",
        "total_tests",
        "mean_throughput_mbps",
        "median_throughput_mbps",
        "p95_throughput_mbps",
        "mean_latency_ms",
        "median_latency_ms",
        "p95_latency_ms",
        "mean_packet_loss_percent",
        "median_packet_loss_percent",
        "p95_packet_loss_percent",
        "lossy_flows",
        "lossy_flow_percent",
    ]

    with open(EVAL_SUMMARY_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for policy, summary in [("FIFO", fifo_summary), ("RL", rl_summary)]:
            row = {"policy": policy}
            row.update(summary)
            writer.writerow(row)

    print(f"Saved summary CSV: {EVAL_SUMMARY_FILE}")


def write_by_seed_summary_csv(fifo_rows, rl_rows):
    fieldnames = [
        "policy",
        "eval_run",
        "eval_seed",
        "total_tests",
        "mean_throughput_mbps",
        "median_throughput_mbps",
        "p95_throughput_mbps",
        "mean_latency_ms",
        "median_latency_ms",
        "p95_latency_ms",
        "mean_packet_loss_percent",
        "median_packet_loss_percent",
        "p95_packet_loss_percent",
        "lossy_flows",
        "lossy_flow_percent",
    ]

    with open(EVAL_BY_SEED_SUMMARY_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for policy, rows in [("FIFO", fifo_rows), ("RL", rl_rows)]:
            for (eval_run, seed), summary in summarize_by_seed(rows).items():
                row = {
                    "policy": policy,
                    "eval_run": eval_run,
                    "eval_seed": seed,
                }
                row.update(summary)
                writer.writerow(row)

    print(f"Saved per-seed summary CSV: {EVAL_BY_SEED_SUMMARY_FILE}")


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

    write_summary_csv(fifo_traffic_summary, rl_traffic_summary)
    write_by_seed_summary_csv(fifo_rows, rl_rows)

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

    plot_summary_bars(fifo_traffic_summary, rl_traffic_summary)

    print()
    print("Saved plots in data/plots/")
    print(f"- {PLOT_PREFIX}_fifo_vs_rl_path_usage.png")
    print(f"- {PLOT_PREFIX}_fifo_vs_rl_throughput.png")
    print(f"- {PLOT_PREFIX}_fifo_vs_rl_latency.png")
    print(f"- {PLOT_PREFIX}_fifo_vs_rl_packet_loss.png")
    print(f"- {PLOT_PREFIX}_latency_summary.png")
    print(f"- {PLOT_PREFIX}_throughput_summary.png")
    print(f"- {PLOT_PREFIX}_packet_loss_summary.png")
    print(f"- {PLOT_PREFIX}_lossy_flow_summary.png")


if __name__ == "__main__":
    main()

import csv
import os
from collections import Counter, defaultdict
from statistics import mean, median

import matplotlib.pyplot as plt

from config import (
    EVAL_BY_SEED_SUMMARY_FILE,
    EVAL_RUNTIME_FILE,
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
                    "eval_run": int(row.get("eval_run", 0)),
                    "eval_seed": int(row.get("eval_seed", 0)),
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

    rows.sort(key=lambda r: (r["eval_run"], r["flow_id"]))
    return rows


def percentile(values, pct):
    if not values:
        return 0.0

    values = sorted(values)
    if len(values) == 1:
        return values[0]

    rank = (len(values) - 1) * (pct / 100.0)
    lower = int(rank)
    upper = min(lower + 1, len(values) - 1)
    weight = rank - lower

    return values[lower] * (1.0 - weight) + values[upper] * weight


def average_metric(rows, metric):
    if not rows:
        return 0.0

    return sum(row[metric] for row in rows) / len(rows)


def summarize_traffic(rows):
    throughput_values = [row["throughput_mbps"] for row in rows]
    latency_values = [row["latency_ms"] for row in rows]
    packet_loss_values = [row["packet_loss_percent"] for row in rows]
    lossy_flows = sum(1 for row in rows if row["packet_loss_percent"] > 0.0)
    total = len(rows)

    return {
        "total_tests": total,
        "average_throughput_mbps": mean(throughput_values) if throughput_values else 0.0,
        "median_throughput_mbps": median(throughput_values) if throughput_values else 0.0,
        "p95_throughput_mbps": percentile(throughput_values, 95),
        "average_latency_ms": mean(latency_values) if latency_values else 0.0,
        "median_latency_ms": median(latency_values) if latency_values else 0.0,
        "p95_latency_ms": percentile(latency_values, 95),
        "average_packet_loss_percent": mean(packet_loss_values) if packet_loss_values else 0.0,
        "median_packet_loss_percent": median(packet_loss_values) if packet_loss_values else 0.0,
        "p95_packet_loss_percent": percentile(packet_loss_values, 95),
        "lossy_flows": lossy_flows,
        "lossy_flow_percent": round((lossy_flows / total) * 100.0, 2) if total else 0.0,
    }


def summarize_by_seed(rows):
    groups = defaultdict(list)

    for row in rows:
        groups[row["eval_seed"]].append(row)

    return {
        seed: summarize_traffic(seed_rows)
        for seed, seed_rows in sorted(groups.items())
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
    print(f"Traffic tests:             {summary['total_tests']}")
    print(f"Average throughput:        {summary['average_throughput_mbps']:.3f} Mbps")
    print(f"Median throughput:         {summary['median_throughput_mbps']:.3f} Mbps")
    print(f"95th pct throughput:       {summary['p95_throughput_mbps']:.3f} Mbps")
    print(f"Average latency:           {summary['average_latency_ms']:.3f} ms")
    print(f"Median latency:            {summary['median_latency_ms']:.3f} ms")
    print(f"95th pct latency:          {summary['p95_latency_ms']:.3f} ms")
    print(f"Average packet loss:       {summary['average_packet_loss_percent']:.3f}%")
    print(f"95th pct packet loss:      {summary['p95_packet_loss_percent']:.3f}%")
    print(f"Lossy flows:               {summary['lossy_flows']} ({summary['lossy_flow_percent']:.2f}%)")


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


def write_summary_csv(fifo_summary, rl_summary):
    fields = [
        "policy",
        "total_tests",
        "average_throughput_mbps",
        "median_throughput_mbps",
        "p95_throughput_mbps",
        "average_latency_ms",
        "median_latency_ms",
        "p95_latency_ms",
        "average_packet_loss_percent",
        "median_packet_loss_percent",
        "p95_packet_loss_percent",
        "lossy_flows",
        "lossy_flow_percent",
    ]

    with open(EVAL_SUMMARY_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for policy, summary in [("FIFO", fifo_summary), ("RL", rl_summary)]:
            row = {"policy": policy}
            row.update(summary)
            writer.writerow(row)

    print(f"Saved summary CSV: {EVAL_SUMMARY_FILE}")


def write_by_seed_summary_csv(fifo_by_seed, rl_by_seed):
    fields = [
        "policy",
        "eval_seed",
        "total_tests",
        "average_throughput_mbps",
        "median_throughput_mbps",
        "p95_throughput_mbps",
        "average_latency_ms",
        "median_latency_ms",
        "p95_latency_ms",
        "average_packet_loss_percent",
        "median_packet_loss_percent",
        "p95_packet_loss_percent",
        "lossy_flows",
        "lossy_flow_percent",
    ]

    with open(EVAL_BY_SEED_SUMMARY_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for policy, by_seed in [("FIFO", fifo_by_seed), ("RL", rl_by_seed)]:
            for seed, summary in by_seed.items():
                row = {"policy": policy, "eval_seed": seed}
                row.update(summary)
                writer.writerow(row)

    print(f"Saved per-seed summary CSV: {EVAL_BY_SEED_SUMMARY_FILE}")


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

    fifo_x = list(range(len(fifo_values)))
    rl_x = list(range(len(rl_values)))

    plt.figure()

    if fifo_values:
        fifo_mean = mean(fifo_values)
        plt.plot(fifo_x, fifo_values, marker="o", label="FIFO")
        plt.axhline(
            fifo_mean,
            linestyle="--",
            linewidth=1.5,
            label=f"FIFO mean = {fifo_mean:.3f}",
        )

    if rl_values:
        rl_mean = mean(rl_values)
        plt.plot(rl_x, rl_values, marker="o", label="RL")
        plt.axhline(
            rl_mean,
            linestyle=":",
            linewidth=1.5,
            label=f"RL mean = {rl_mean:.3f}",
        )

    plt.xlabel("Evaluation sample index")
    plt.ylabel(ylabel)
    plt.title(f"{title} ({TOPO_MODE})")
    plt.legend()

    output = os.path.join(PLOTS_DIR, f"{PLOT_PREFIX}_{filename}")
    plt.savefig(output, bbox_inches="tight")
    plt.close()


def plot_summary_bar(metric_key, ylabel, title, filename, fifo_summary, rl_summary):
    labels = ["FIFO", "RL"]
    values = [fifo_summary[metric_key], rl_summary[metric_key]]

    plt.figure()
    plt.bar(labels, values)
    plt.ylabel(ylabel)
    plt.title(f"{title} ({TOPO_MODE})")

    output = os.path.join(PLOTS_DIR, f"{PLOT_PREFIX}_{filename}")
    plt.savefig(output, bbox_inches="tight")
    plt.close()


def plot_seed_metric(fifo_by_seed, rl_by_seed, metric_key, ylabel, title, filename):
    seeds = sorted(set(fifo_by_seed.keys()) | set(rl_by_seed.keys()))
    x = range(len(seeds))

    fifo_values = [fifo_by_seed.get(seed, {}).get(metric_key, 0.0) for seed in seeds]
    rl_values = [rl_by_seed.get(seed, {}).get(metric_key, 0.0) for seed in seeds]

    plt.figure(figsize=(max(8, len(seeds) * 1.2), 5))
    plt.bar([i - 0.2 for i in x], fifo_values, width=0.4, label="FIFO")
    plt.bar([i + 0.2 for i in x], rl_values, width=0.4, label="RL")
    plt.xticks(list(x), [str(seed) for seed in seeds], rotation=45)
    plt.xlabel("Evaluation seed")
    plt.ylabel(ylabel)
    plt.title(f"{title} by Seed ({TOPO_MODE})")
    plt.legend()
    plt.tight_layout()

    output = os.path.join(PLOTS_DIR, f"{PLOT_PREFIX}_{filename}")
    plt.savefig(output, bbox_inches="tight")
    plt.close()


def read_runtime_rows():
    if not os.path.exists(EVAL_RUNTIME_FILE):
        return []

    with open(EVAL_RUNTIME_FILE, "r") as f:
        return list(csv.DictReader(f))


def print_runtime_summary():
    rows = read_runtime_rows()

    if not rows:
        print(f"[WARN] Missing evaluation runtime file: {EVAL_RUNTIME_FILE}")
        return

    total_rows = [row for row in rows if row.get("eval_run") == "ALL"]
    trace_rows = [row for row in rows if row.get("eval_run") != "ALL"]

    print()
    print(f"Evaluation Runtime ({TOPO_MODE})")
    print("-" * 50)

    for row in total_rows:
        print(
            f"{row.get('policy', '')} full test: "
            f"{row.get('duration_human', '')} "
            f"({row.get('duration_seconds', '')} seconds)"
        )

    durations = []

    for row in trace_rows:
        try:
            durations.append(float(row.get("duration_seconds", 0.0)))
        except ValueError:
            pass

    if durations:
        print(f"Average trace runtime: {mean(durations):.2f} seconds")



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

    fifo_by_seed = summarize_by_seed(fifo_rows)
    rl_by_seed = summarize_by_seed(rl_rows)

    print_path_summary("FIFO", fifo_path_summary)
    print_path_summary("RL", rl_path_summary)

    print_traffic_summary("FIFO", fifo_traffic_summary)
    print_traffic_summary("RL", rl_traffic_summary)
    print_runtime_summary()

    validate_counts("FIFO", fifo_decisions, fifo_rows)
    validate_counts("RL", rl_decisions, rl_rows)

    write_summary_csv(fifo_traffic_summary, rl_traffic_summary)
    write_by_seed_summary_csv(fifo_by_seed, rl_by_seed)

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

    plot_summary_bar(
        "average_latency_ms",
        "Latency (ms)",
        "Average Latency",
        "summary_average_latency.png",
        fifo_traffic_summary,
        rl_traffic_summary,
    )

    plot_summary_bar(
        "p95_latency_ms",
        "Latency (ms)",
        "95th Percentile Latency",
        "summary_p95_latency.png",
        fifo_traffic_summary,
        rl_traffic_summary,
    )

    plot_summary_bar(
        "average_throughput_mbps",
        "Throughput (Mbps)",
        "Average Throughput",
        "summary_average_throughput.png",
        fifo_traffic_summary,
        rl_traffic_summary,
    )

    plot_summary_bar(
        "lossy_flow_percent",
        "Lossy Flows (%)",
        "Lossy Flow Percentage",
        "summary_lossy_flow_percent.png",
        fifo_traffic_summary,
        rl_traffic_summary,
    )

    plot_seed_metric(
        fifo_by_seed,
        rl_by_seed,
        "average_latency_ms",
        "Average Latency (ms)",
        "Average Latency",
        "seed_average_latency.png",
    )

    plot_seed_metric(
        fifo_by_seed,
        rl_by_seed,
        "average_throughput_mbps",
        "Average Throughput (Mbps)",
        "Average Throughput",
        "seed_average_throughput.png",
    )

    plot_seed_metric(
        fifo_by_seed,
        rl_by_seed,
        "lossy_flow_percent",
        "Lossy Flows (%)",
        "Lossy Flow Percentage",
        "seed_lossy_flow_percent.png",
    )

    print()
    print("Saved plots in data/plots/")
    print(f"Saved summaries: {EVAL_SUMMARY_FILE}, {EVAL_BY_SEED_SUMMARY_FILE}")


if __name__ == "__main__":
    main()

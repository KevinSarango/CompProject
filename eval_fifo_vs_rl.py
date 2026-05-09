import csv
import os
from collections import Counter

import matplotlib.pyplot as plt


DATA_DIR = "data"
PLOTS_DIR = "data/plots"

FIFO_DECISIONS = os.path.join(DATA_DIR, "fifo_metrics.csv")
RL_DECISIONS = os.path.join(DATA_DIR, "rl_metrics.csv")

FIFO_TRAFFIC = os.path.join(DATA_DIR, "fifo_traffic_metrics.csv")
RL_TRAFFIC = os.path.join(DATA_DIR, "rl_traffic_metrics.csv")


def ensure_dirs():
    os.makedirs(PLOTS_DIR, exist_ok=True)


def summarize_path_usage(path):
    if not os.path.exists(path):
        return None

    paths = []

    with open(path, "r") as f:
        reader = csv.DictReader(f)

        for row in reader:
            if "path" in row:
                paths.append(row["path"])

    counts = Counter(paths)
    total = sum(counts.values())

    if total == 0:
        return {
            "total_flows": 0,
            "upper": 0,
            "lower": 0,
            "upper_percent": 0,
            "lower_percent": 0,
        }

    return {
        "total_flows": total,
        "upper": counts["upper"],
        "lower": counts["lower"],
        "upper_percent": round((counts["upper"] / total) * 100, 2),
        "lower_percent": round((counts["lower"] / total) * 100, 2),
    }


def read_traffic_metrics(path):
    if not os.path.exists(path):
        return []

    rows = []

    with open(path, "r") as f:
        reader = csv.DictReader(f)

        for row in reader:
            rows.append({
                "policy": row["policy"],
                "src": row["src"],
                "dst": row["dst"],
                "throughput_mbps": float(row["throughput_mbps"]),
                "latency_ms": float(row["latency_ms"]),
                "packet_loss_percent": float(row["packet_loss_percent"]),
            })

    return rows


def average_metric(rows, metric):
    if not rows:
        return 0.0

    return sum(row[metric] for row in rows) / len(rows)


def summarize_traffic(rows):
    return {
        "average_throughput_mbps": average_metric(rows, "throughput_mbps"),
        "average_latency_ms": average_metric(rows, "latency_ms"),
        "average_packet_loss_percent": average_metric(rows, "packet_loss_percent"),
    }


def print_path_summary(name, summary):
    if summary is None:
        print(f"\n{name} path decision file missing.")
        return

    print()
    print(f"{name} Path Usage")
    print("-" * 35)
    print(f"Total flows:   {summary['total_flows']}")
    print(f"Upper path:    {summary['upper']} ({summary['upper_percent']}%)")
    print(f"Lower path:    {summary['lower']} ({summary['lower_percent']}%)")


def print_traffic_summary(name, summary):
    print()
    print(f"{name} Traffic Metrics")
    print("-" * 35)
    print(f"Average throughput:  {summary['average_throughput_mbps']:.3f} Mbps")
    print(f"Average latency:     {summary['average_latency_ms']:.3f} ms")
    print(f"Average packet loss: {summary['average_packet_loss_percent']:.3f}%")


def plot_path_usage(fifo_summary, rl_summary):
    if fifo_summary is None or rl_summary is None:
        return

    labels = ["Upper Path", "Lower Path"]
    fifo_values = [fifo_summary["upper"], fifo_summary["lower"]]
    rl_values = [rl_summary["upper"], rl_summary["lower"]]

    x = range(len(labels))

    plt.figure()

    plt.bar(
        [i - 0.2 for i in x],
        fifo_values,
        width=0.4,
        label="FIFO",
    )

    plt.bar(
        [i + 0.2 for i in x],
        rl_values,
        width=0.4,
        label="RL",
    )

    plt.xticks(list(x), labels)
    plt.ylabel("Number of Flows")
    plt.title("FIFO vs RL Path Usage")
    plt.legend()

    plt.savefig(
        os.path.join(PLOTS_DIR, "fifo_vs_rl_path_usage.png"),
        bbox_inches="tight",
    )

    plt.close()


def plot_metric_line_graph(
    fifo_rows,
    rl_rows,
    metric,
    ylabel,
    title,
    filename,
):
    fifo_values = [row[metric] for row in fifo_rows]
    rl_values = [row[metric] for row in rl_rows]

    max_len = max(len(fifo_values), len(rl_values), 1)

    fifo_x = list(range(1, len(fifo_values) + 1))
    rl_x = list(range(1, len(rl_values) + 1))

    plt.figure()

    if fifo_values:
        plt.plot(
            fifo_x,
            fifo_values,
            marker="o",
            label="FIFO",
        )

    if rl_values:
        plt.plot(
            rl_x,
            rl_values,
            marker="o",
            label="RL",
        )

    plt.xlabel("Traffic Flow Test")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(range(1, max_len + 1))
    plt.legend()

    plt.savefig(
        os.path.join(PLOTS_DIR, filename),
        bbox_inches="tight",
    )

    plt.close()


def main():
    ensure_dirs()

    fifo_path_summary = summarize_path_usage(FIFO_DECISIONS)
    rl_path_summary = summarize_path_usage(RL_DECISIONS)

    fifo_rows = read_traffic_metrics(FIFO_TRAFFIC)
    rl_rows = read_traffic_metrics(RL_TRAFFIC)

    fifo_traffic_summary = summarize_traffic(fifo_rows)
    rl_traffic_summary = summarize_traffic(rl_rows)

    print_path_summary("FIFO", fifo_path_summary)
    print_path_summary("RL", rl_path_summary)

    print_traffic_summary("FIFO", fifo_traffic_summary)
    print_traffic_summary("RL", rl_traffic_summary)

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
    print("Saved plots:")
    print("- data/plots/fifo_vs_rl_path_usage.png")
    print("- data/plots/fifo_vs_rl_throughput.png")
    print("- data/plots/fifo_vs_rl_latency.png")
    print("- data/plots/fifo_vs_rl_packet_loss.png")


if __name__ == "__main__":
    main()

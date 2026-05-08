import csv
import os
from collections import Counter


def summarize(path):
    if not os.path.exists(path):
        print(f"Missing file: {path}")
        return None

    paths = []

    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
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


def print_summary(name, summary):
    if summary is None:
        return

    print()
    print(f"{name} Summary")
    print("-" * 30)
    print(f"Total flows:   {summary['total_flows']}")
    print(f"Upper path:    {summary['upper']} ({summary['upper_percent']}%)")
    print(f"Lower path:    {summary['lower']} ({summary['lower_percent']}%)")


if __name__ == "__main__":
    fifo = summarize("data/fifo_metrics.csv")
    rl = summarize("data/rl_metrics.csv")

    print_summary("FIFO", fifo)
    print_summary("RL", rl)

    print()
    print("Use this in the report:")
    print("FIFO is the baseline flow assignment policy.")
    print("RL uses an offline-trained Q-table to choose between paths.")

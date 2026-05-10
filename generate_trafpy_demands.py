import csv
import os
import random


OUTPUT_FILE = "data/trafpy_demands.csv"

TOP_HOSTS = ["h1", "h2", "h3", "h4"]
BOTTOM_HOSTS = ["h5", "h6", "h7", "h8"]


def generate_demands(num_flows=80, seed=42):
    random.seed(seed)

    demands = []
    current_time = 0.0

    for flow_id in range(num_flows):
        if flow_id % 2 == 0:
            src = random.choice(TOP_HOSTS)
            dst = random.choice(BOTTOM_HOSTS)
        else:
            src = random.choice(BOTTOM_HOSTS)
            dst = random.choice(TOP_HOSTS)

        # Bursty arrivals. Many flows start close together.
        current_time += random.choice([0.0, 0.0, 0.02, 0.05, 0.1])

        # Flow size is now in KB, not MB.
        size_kb = round(random.uniform(100.0, 1000.0), 1)

        demands.append({
            "flow_id": flow_id,
            "src": src,
            "dst": dst,
            "start_time": round(current_time, 2),
            "size_kb": size_kb,
        })

    return demands


def save_demands(demands):
    os.makedirs("data", exist_ok=True)

    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["flow_id", "src", "dst", "start_time", "size_kb"],
        )
        writer.writeheader()
        writer.writerows(demands)

    print(f"Saved {len(demands)} demands to {OUTPUT_FILE}")


if __name__ == "__main__":
    demands = generate_demands(num_flows=80)
    save_demands(demands)

    print()
    print("First 10 demands:")
    for demand in demands[:10]:
        print(demand)

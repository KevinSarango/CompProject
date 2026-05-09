import csv
import json
import os

import matplotlib.pyplot as plt


PLOTS_DIR = "data/plots"


def ensure_directories():
    os.makedirs(PLOTS_DIR, exist_ok=True)


def read_rewards():
    episodes = []
    total_rewards = []
    average_rewards = []

    with open("data/training_rewards.csv", "r") as f:
        reader = csv.DictReader(f)

        for row in reader:
            episodes.append(int(row["episode"]))
            total_rewards.append(float(row["total_reward"]))
            average_rewards.append(float(row["average_reward"]))

    return episodes, total_rewards, average_rewards


def plot_rewards():
    episodes, total_rewards, average_rewards = read_rewards()

    plt.figure()
    plt.plot(episodes, total_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("RL Training: Total Reward per Episode")

    total_path = os.path.join(PLOTS_DIR, "reward_curve_total.png")

    plt.savefig(total_path, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(episodes, average_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.title("RL Training: Average Reward per Episode")

    average_path = os.path.join(PLOTS_DIR, "reward_curve_average.png")

    plt.savefig(average_path, bbox_inches="tight")
    plt.close()


def plot_q_table():
    with open("data/q_table.json", "r") as f:
        q_table = json.load(f)

    states = sorted(q_table.keys(), key=int)

    upper_values = [q_table[s]["0"] for s in states]
    lower_values = [q_table[s]["1"] for s in states]

    x = range(len(states))

    plt.figure()

    plt.bar(
        [i - 0.2 for i in x],
        upper_values,
        width=0.4,
        label="upper path",
    )

    plt.bar(
        [i + 0.2 for i in x],
        lower_values,
        width=0.4,
        label="lower path",
    )

    plt.xticks(list(x), [f"state {s}" for s in states])

    plt.xlabel("State")
    plt.ylabel("Q-value")
    plt.title("Learned Q-table Values")

    plt.legend()

    qtable_path = os.path.join(PLOTS_DIR, "q_table_values.png")

    plt.savefig(qtable_path, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    ensure_directories()

    plot_rewards()
    plot_q_table()

    print("Saved plots:")
    print(f"- {PLOTS_DIR}/reward_curve_total.png")
    print(f"- {PLOTS_DIR}/reward_curve_average.png")
    print(f"- {PLOTS_DIR}/q_table_values.png")

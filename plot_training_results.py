import csv
import json
import os

import matplotlib.pyplot as plt


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
    plt.savefig("data/reward_curve_total.png", bbox_inches="tight")

    plt.figure()
    plt.plot(episodes, average_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.title("RL Training: Average Reward per Episode")
    plt.savefig("data/reward_curve_average.png", bbox_inches="tight")


def plot_q_table():
    with open("data/q_table.json", "r") as f:
        q_table = json.load(f)

    states = sorted(q_table.keys(), key=int)
    upper_values = [q_table[s]["0"] for s in states]
    lower_values = [q_table[s]["1"] for s in states]

    x = range(len(states))

    plt.figure()
    plt.bar([i - 0.2 for i in x], upper_values, width=0.4, label="upper")
    plt.bar([i + 0.2 for i in x], lower_values, width=0.4, label="lower")
    plt.xticks(list(x), [f"state {s}" for s in states])
    plt.xlabel("State")
    plt.ylabel("Q-value")
    plt.title("Learned Q-table Values")
    plt.legend()
    plt.savefig("data/q_table_values.png", bbox_inches="tight")


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    plot_rewards()
    plot_q_table()

    print("Saved graphs:")
    print("- data/reward_curve_total.png")
    print("- data/reward_curve_average.png")
    print("- data/q_table_values.png")

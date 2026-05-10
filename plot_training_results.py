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
    plt.savefig(os.path.join(PLOTS_DIR, "reward_curve_total.png"), bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(episodes, average_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.title("RL Training: Average Reward per Episode")
    plt.savefig(os.path.join(PLOTS_DIR, "reward_curve_average.png"), bbox_inches="tight")
    plt.close()


def plot_q_table_policy():
    """
    For 18 states, plotting every Q-value as a grouped bar chart can be too dense.
    This plot shows the learned best action for each state.
    """
    with open("data/q_table.json", "r") as f:
        q_table = json.load(f)

    states = sorted(
        q_table.keys(),
        key=lambda s: tuple(int(part) for part in s.split("_")),
    )

    best_actions = []

    for state in states:
        values = q_table[state]
        best_action = int(max(values, key=values.get))
        best_actions.append(best_action)

    x = range(len(states))

    plt.figure(figsize=(12, 5))
    plt.plot(list(x), best_actions, marker="o")
    plt.yticks([0, 1], ["upper", "lower"])
    plt.xticks(list(x), states, rotation=90)
    plt.xlabel("State: utilization_bin_demand_bin_previous_action")
    plt.ylabel("Best Action")
    plt.title("Learned Policy from 18-State Q-table")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "q_table_policy.png"), bbox_inches="tight")
    plt.close()


def plot_q_table_values():
    with open("data/q_table.json", "r") as f:
        q_table = json.load(f)

    states = sorted(
        q_table.keys(),
        key=lambda s: tuple(int(part) for part in s.split("_")),
    )

    upper_values = [q_table[s]["0"] for s in states]
    lower_values = [q_table[s]["1"] for s in states]

    x = range(len(states))

    plt.figure(figsize=(14, 6))
    plt.bar([i - 0.2 for i in x], upper_values, width=0.4, label="upper path")
    plt.bar([i + 0.2 for i in x], lower_values, width=0.4, label="lower path")
    plt.xticks(list(x), states, rotation=90)
    plt.xlabel("State: utilization_bin_demand_bin_previous_action")
    plt.ylabel("Q-value")
    plt.title("18-State Q-table Values")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "q_table_values.png"), bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    ensure_directories()
    plot_rewards()
    plot_q_table_values()
    plot_q_table_policy()

    print("Saved plots:")
    print(f"- {PLOTS_DIR}/reward_curve_total.png")
    print(f"- {PLOTS_DIR}/reward_curve_average.png")
    print(f"- {PLOTS_DIR}/q_table_values.png")
    print(f"- {PLOTS_DIR}/q_table_policy.png")

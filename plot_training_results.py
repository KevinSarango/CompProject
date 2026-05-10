import csv
import json
import os

import matplotlib.pyplot as plt

from config import ACTION_TO_PATH, PATHS, PLOTS_DIR, PLOT_PREFIX, Q_TABLE_FILE, TRAINING_REWARDS_FILE


def ensure_directories():
    os.makedirs(PLOTS_DIR, exist_ok=True)


def read_rewards():
    episodes = []
    total_rewards = []
    average_rewards = []

    with open(TRAINING_REWARDS_FILE, "r") as f:
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
    plt.title(f"RL Training: Total Reward per Episode ({PLOT_PREFIX})")
    plt.savefig(
        os.path.join(PLOTS_DIR, f"{PLOT_PREFIX}_reward_curve_total.png"),
        bbox_inches="tight",
    )
    plt.close()

    plt.figure()
    plt.plot(episodes, average_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.title(f"RL Training: Average Reward per Episode ({PLOT_PREFIX})")
    plt.savefig(
        os.path.join(PLOTS_DIR, f"{PLOT_PREFIX}_reward_curve_average.png"),
        bbox_inches="tight",
    )
    plt.close()


def sorted_states(q_table):
    return sorted(
        q_table.keys(),
        key=lambda s: tuple(int(part) for part in s.split("_")),
    )


def plot_q_table_policy():
    with open(Q_TABLE_FILE, "r") as f:
        q_table = json.load(f)

    states = sorted_states(q_table)
    best_actions = []

    for state in states:
        values = q_table[state]
        best_action = int(max(values, key=values.get))
        best_actions.append(best_action)

    x = range(len(states))

    plt.figure(figsize=(12, 5))
    plt.plot(list(x), best_actions, marker="o")
    plt.yticks(
        list(range(len(PATHS))),
        [ACTION_TO_PATH[str(i)] for i in range(len(PATHS))],
    )
    plt.xticks(list(x), states, rotation=90)
    plt.xlabel("State: least_utilized_path_demand_bin_previous_action")
    plt.ylabel("Best Action")
    plt.title(f"Learned Policy from Q-table ({PLOT_PREFIX})")
    plt.tight_layout()
    plt.savefig(
        os.path.join(PLOTS_DIR, f"{PLOT_PREFIX}_q_table_policy.png"),
        bbox_inches="tight",
    )
    plt.close()


def plot_q_table_values():
    with open(Q_TABLE_FILE, "r") as f:
        q_table = json.load(f)

    states = sorted_states(q_table)
    x = range(len(states))

    plt.figure(figsize=(14, 6))

    width = 0.8 / len(PATHS)

    for action_index, path_name in enumerate(PATHS):
        values = [q_table[s][str(action_index)] for s in states]
        offsets = [i - 0.4 + width / 2 + action_index * width for i in x]

        plt.bar(
            offsets,
            values,
            width=width,
            label=path_name,
        )

    plt.xticks(list(x), states, rotation=90)
    plt.xlabel("State: least_utilized_path_demand_bin_previous_action")
    plt.ylabel("Q-value")
    plt.title(f"Q-table Values ({PLOT_PREFIX})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(PLOTS_DIR, f"{PLOT_PREFIX}_q_table_values.png"),
        bbox_inches="tight",
    )
    plt.close()


if __name__ == "__main__":
    ensure_directories()
    plot_rewards()
    plot_q_table_values()
    plot_q_table_policy()

    print("Saved plots:")
    print(f"- {PLOTS_DIR}/{PLOT_PREFIX}_reward_curve_total.png")
    print(f"- {PLOTS_DIR}/{PLOT_PREFIX}_reward_curve_average.png")
    print(f"- {PLOTS_DIR}/{PLOT_PREFIX}_q_table_values.png")
    print(f"- {PLOTS_DIR}/{PLOT_PREFIX}_q_table_policy.png")
